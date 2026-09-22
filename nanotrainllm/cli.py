"""The entry point: argparse -> `Config` -> `Trainer`.  Six stages, one file.

    python -m nanotrainllm.cli --stage sft --data examples/data/sft.jsonl
    torchrun --nproc-per-node 8 -m nanotrainllm.cli --stage sft --fsdp2 ...

**The flags are generated from `Config`'s fields**, so a flag cannot drift away
from the knob it sets.  The `DERIVED` sentinels get none: they come from the
checkpoint or the launcher, where a flag could only contradict reality.

**It decides who gets built** -- the whole collaborator table:

    stage    teacher   ref                              rollout
    pt/sft   -         -                                -
    dpo      -         only if --tuner full (else       -
                       Policy.no_adapter() is pi_ref)
    grpo     -         only if --kl-coef != 0           yes
    opd      yes       -                                if --on-policy-ratio > 0
    opd-rl   yes       only if --kl-coef != 0           yes

So `stages/` can assert its collaborators exist instead of testing at every use.

**`torch.manual_seed(cfg.seed)` is rank-*independent*.**  `fully_shard` builds the
whole model on every rank and keeps one slice, so ranks whose LoRA init disagreed
would assemble a model out of slices of different draws.
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import fields

import torch

from .config import Config
from .dataset import Dataset
from .loop import Trainer
from .template import Template

# Sentinels: filled in by `Config.bind_model_meta` or by the launcher.
DERIVED = frozenset(
    {
        "hf_config",
        "vocab_size",
        "image_token_id",
        "merge_size",
        "pad_token_id",
        "eos_token_id",
        "world_size",
        "rank",
    }
)
# `from __future__ import annotations` in config.py makes `f.type` a string, so this
# table matches on source text -- an unlisted annotation raises rather than guessing.
SCALARS = {"str": str, "str | None": str, "int": int, "float": float}


def parser() -> argparse.ArgumentParser:
    """One flag per `Config` field, derived from the dataclass itself."""
    p = argparse.ArgumentParser(prog="nanotrainllm", description=__doc__.splitlines()[0])
    for f in fields(Config):
        if f.name in DERIVED:
            continue
        flag = "--" + f.name.replace("_", "-")
        if f.type == "bool":
            # BooleanOptionalAction is where `--no-vllm-sleep` comes from.
            p.add_argument(flag, default=f.default, action=argparse.BooleanOptionalAction)
        elif f.type in SCALARS:
            p.add_argument(flag, default=f.default, type=SCALARS[f.type])
        else:
            assert f.name == "betas", f"no flag rule for {f.name}: {f.type}"
            p.add_argument(flag, default=list(f.default), type=float, nargs=2)
    return p


def apply_dry_run(cfg: Config) -> None:
    """CPU, random weights, two steps: all of the plumbing and none of the cost.

    `tiny_model` shrinks depth and width but keeps the *real* vocabulary and token
    ids, so the batch that flows through is the batch a real run would see.  That is
    what makes this the right way to check a new dataset, reward or template -- not a
    debug hook.  Two combinations it cannot reach: `--rollout vllm` needs a GPU, and
    `--packing` / `--padding-free` need a varlen flash-attention kernel.
    """
    assert cfg.rollout == "hf", "--dry-run is CPU-only; the vllm engine needs a GPU"
    cfg.tiny_model = True
    cfg.attn_impl = "sdpa"
    cfg.gradient_checkpointing = False
    cfg.fsdp2 = False
    if cfg.max_steps < 0:
        cfg.max_steps = 2


def init_distributed(cfg: Config) -> str:
    """Fill in `cfg.rank` / `cfg.world_size` from the environment; -> device string.

    `dataset.batches` shards on these two numbers, so they have to be set before
    the `Dataset` is built, not before the first step.
    """
    cfg.rank = int(os.environ.get("RANK", 0))
    cfg.world_size = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if cfg.dry_run or not torch.cuda.is_available():
        assert cfg.world_size == 1, "the CPU path is single-process; drop torchrun"
        return "cpu"
    torch.cuda.set_device(local)
    if cfg.world_size > 1:
        import torch.distributed as dist

        dist.init_process_group("nccl")
    return f"cuda:{local}"


def build_all(cfg: Config, device: str):
    """-> `(dataset, policy, collaborators)`, in the order the pieces depend on.

    The dataset comes before the model because encoding is where a bad row raises;
    finding that out before the weights are on the card saves a minute every time.
    """
    from transformers import AutoConfig, AutoProcessor

    from .model import load_policy, load_ref, load_teacher

    # `max_pixels` caps each image's h*w *after* smart-resize, so a large figure costs
    # fewer vision tokens; -1 leaves the processor's own default alone.
    proc_kwargs = {"max_pixels": cfg.max_pixels} if cfg.max_pixels > 0 else {}
    processor = AutoProcessor.from_pretrained(cfg.model, **proc_kwargs)
    cfg.bind_model_meta(AutoConfig.from_pretrained(cfg.model), processor.tokenizer)
    template = Template(processor, cfg)
    dataset = Dataset(cfg.data, template, cfg)

    policy = load_policy(cfg, device)
    if cfg.fsdp2:
        from .model import shard_fsdp2

        shard_fsdp2(policy.model, cfg)

    kw: dict = {}
    if cfg.teacher:
        kw["teacher"] = load_teacher(cfg, device)
        if cfg.fsdp2:
            shard_fsdp2(kw["teacher"].model, cfg)
    if cfg.needs_ref_model and not (cfg.stage == "dpo" and cfg.tuner == "lora"):
        kw["ref"] = load_ref(cfg, device)
    if cfg.needs_rollout:
        from .rl.rollout import build_rollout

        kw["rollout"] = build_rollout(cfg, policy, processor.tokenizer)
    return dataset, policy, kw


def report(cfg: Config, dataset: Dataset, collaborators: dict) -> None:
    """One screen of what is about to run, printed by rank 0 and nothing else.

    No logging framework: a library that configures logging is a library you cannot
    embed.  `Trainer.run()` returns its history as data instead.
    """
    s = dataset.stats()
    print(f"nano-trainllm  stage={cfg.stage}  tuner={cfg.tuner}  seed={cfg.seed}")
    print(f"  model     : {cfg.model}" + ("  (tiny random weights)" if cfg.tiny_model else ""))
    if cfg.teacher:
        print(f"  teacher   : {cfg.teacher}")
    print(f"  data      : {cfg.data}")
    print(
        f"  dataset   : {s['groups']} groups / {s['samples']} samples, dropped {s['dropped']}, "
        f"{s['tokens']} tokens ({s['supervised_frac']:.1%} supervised, "
        f"{s['image_tokens']} vision), len p50/p99/max {s['len_p50']}/{s['len_p99']}/{s['len_max']}"
    )
    layout = "packed" if cfg.packing else ("padding-free" if cfg.padding_free else "padded")
    print(
        f"  batch     : {cfg.micro_batch_size} x accum {cfg.grad_accum} x world "
        f"{cfg.world_size} = {cfg.effective_batch_size} ({layout})"
    )
    if cfg.is_on_policy:
        # For a rollout stage the three-tier schedule, not micro x accum, sets the step
        # count -- and >1 replay means the clipped ratio is live.
        print(
            f"  rollout   : {cfg.rollout_prompts} prompts / global {cfg.global_prompts} "
            f"= {cfg.updates_per_rollout} updates x {cfg.num_iterations} ppo-epochs "
            f"x G {cfg.num_generations}  (forward {cfg.micro_batch_size} completions)"
        )
    print(f"  extras    : {sorted(collaborators) or 'none'}")
    print(f"  output    : {cfg.output_dir}")


def main(argv: list[str] | None = None) -> list[dict]:
    """-> the step-by-step history, so a script or a notebook can assert on it."""
    cfg = Config.from_kwargs(**vars(parser().parse_args(argv)))
    assert cfg.data, "--data is required"
    if cfg.dry_run:
        apply_dry_run(cfg)
    device = init_distributed(cfg)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    dataset, policy, collaborators = build_all(cfg, device)
    if cfg.rank == 0:
        report(cfg, dataset, collaborators)
    history = Trainer(cfg, policy, dataset, **collaborators).run()
    if cfg.rank == 0 and history:
        last = history[-1]
        print(f"done: {len(history)} steps, final loss {last['loss']:.4f} -> {cfg.output_dir}")
    return history


if __name__ == "__main__":
    main()
