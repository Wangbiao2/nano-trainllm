"""Fold a trained checkpoint back into the base model and write a standalone HF dir.

A `step-N.pt` is not a deployable model.  For a LoRA run it is *only* the adapter
factors (`Policy.state_dict` filters to `lora_` keys); for `--tuner full` it is the
trained parameters but still no config, no tokenizer, no vision buffers.

    python tools/export_merged.py \\
        --checkpoint output/run/step-240.pt --output output/run/merged

Why it reuses the trainer's own code rather than re-deriving the fold:

  * `W += (alpha / r) * B @ A` is exactly `merged_full_state` from `model.py` -- the
    same generator vLLM syncs through -- run on a plain CPU model, so the exported
    weights are bit-for-bit what a rollout would have seen.  Not
    `Policy.merged_adapter()`: its peft `merge()`/`unmerge()` round trip is
    `(W + D) - D`, not bit-exact in bf16.
  * The clean model is built with `adapters=False` and the merged tensors copied in, so
    every buffer the checkpoint never stored -- rotary `inv_freq`, the vision tower's
    registered tensors -- comes from the base checkpoint, and the
    `lm_head`/`embed_tokens` tie survives the in-place `copy_`.
  * `rank` and the target modules are read off the checkpoint's key shapes; only
    `--lora-alpha` cannot be (a scalar leaves no trace), so it must match training.

Not for FSDP2 shards: `Policy.state_dict` already `full_tensor()`-gathers, so this runs
single-process on CPU and needs no launcher.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nanotrainllm.config import Config  # noqa: E402
from nanotrainllm.model import build_model, merged_full_state  # noqa: E402

LORA_A_SUFFIX = ".lora_A.default.weight"


def is_lora_ckpt(sd: dict) -> bool:
    return any("lora_" in k for k in sd)


def derive_lora(sd: dict) -> tuple[int, list[str]]:
    """(rank, target module names) from the checkpoint's own keys.

    `lora_A.default.weight` is `[r, in_features]`, so its first dim is the rank; the
    module it belongs to is the path segment just before `.lora_A`.  A consistent run
    has one rank across every adapter, so a disagreement is a corrupt checkpoint.
    """
    ranks, targets = set(), set()
    for k, v in sd.items():
        if k.endswith(LORA_A_SUFFIX):
            ranks.add(v.shape[0])
            targets.add(k[: -len(LORA_A_SUFFIX)].rsplit(".", 1)[1])
    assert targets, f"no {LORA_A_SUFFIX!r} keys; is this a LoRA checkpoint?"
    assert len(ranks) == 1, f"inconsistent lora rank across adapters: {sorted(ranks)}"
    return ranks.pop(), sorted(targets)


def make_cfg(model: str, tuner: str, rank: int, alpha: int, targets: list[str]) -> Config:
    """A CPU/eval config for the merge.

    `dry_run=True` is the one non-obvious flag: `Config.__post_init__` asserts a real
    `--data` file exists unless `dry_run` or `tiny_model` is set, and neither the base
    weights nor the checkpoint say anything about training data.  It does *not* pull in
    `apply_dry_run` (that shrinks the model); it only relaxes the asserts.
    """
    return Config.from_kwargs(
        model=model,
        tuner=tuner,
        lora_rank=rank,
        lora_alpha=alpha,
        lora_target=",".join(targets),
        attn_impl="sdpa",  # flash-attn has no CPU kernel
        gradient_checkpointing=False,
        fsdp2=False,
        dry_run=True,
    )


def check_load(missing: list[str], unexpected: list[str], what: str) -> None:
    """`load_state_dict(strict=False)` swallows both lists; a checkpoint that does not
    fit the model is a silent wrong export, so surface either as a hard error."""
    assert not unexpected, f"{what}: {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}"
    assert not missing, f"{what}: {len(missing)} missing keys, e.g. {missing[:3]}"


def export_lora(sd: dict, model: str, alpha: int, device: str) -> torch.nn.Module:
    rank, targets = derive_lora(sd)
    print(f"  lora      : rank={rank}  alpha={alpha}  targets={targets}")
    cfg = make_cfg(model, "lora", rank, alpha, targets)

    # Inject the same adapters training used, load the deltas, fold them on full tensors.
    inj = build_model(cfg).to(device)
    res = inj.load_state_dict(sd, strict=False)
    # The base weights are legitimately "missing" from a LoRA checkpoint -- it stores
    # only `lora_` keys -- so only unexpected keys can indict the checkpoint here.
    assert not res.unexpected_keys, (
        f"adapter load: unexpected keys, e.g. {res.unexpected_keys[:3]}"
    )
    lora_missing = [k for k in res.missing_keys if "lora_" in k]
    assert not lora_missing, (
        f"adapter load: {len(lora_missing)} lora keys unfilled, e.g. {lora_missing[:3]}"
    )
    merged = dict(merged_full_state(inj))
    del inj

    clean = build_model(cfg, adapters=False).to(device)
    res = clean.load_state_dict(merged, strict=False)
    check_load(res.missing_keys, res.unexpected_keys, "merged load")
    return clean


def export_full(sd: dict, model: str, device: str) -> torch.nn.Module:
    print("  full      : no adapters, loading trained weights directly")
    cfg = make_cfg(model, "full", 1, 32, ["q_proj"])  # rank/alpha/targets unused for full
    clean = build_model(cfg, adapters=False).to(device)
    res = clean.load_state_dict(sd, strict=False)
    check_load(res.missing_keys, res.unexpected_keys, "full load")
    return clean


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="export_merged", description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--checkpoint", required=True, help="a step-N.pt written by Trainer.save")
    p.add_argument("--output", required=True, help="directory to write the merged HF model")
    p.add_argument("--lora-alpha", type=int, default=32, help="must match training")
    p.add_argument("--device", default="cpu", help="cpu keeps GPUs free; it is a few matmuls")
    args = p.parse_args(argv)

    assert os.path.isfile(args.checkpoint), f"not a file: {args.checkpoint}"
    assert os.path.isdir(args.model) or "/" in args.model, f"no such model: {args.model!r}"

    print(f"export_merged  {args.checkpoint} -> {args.output}")
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    if is_lora_ckpt(sd):
        model = export_lora(sd, args.model, args.lora_alpha, args.device)
    else:
        model = export_full(sd, args.model, args.device)

    # Undo the training-time settings before writing: a deployable model wants its KV
    # cache and its weight tie back (`build_model` sets use_cache=False for training).
    model.tie_weights()
    model.config.use_cache = True
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)

    # The processor is the other half of a usable checkpoint: tokenizer + image config.
    from transformers import AutoProcessor

    AutoProcessor.from_pretrained(args.model).save_pretrained(args.output)
    print(f"done: merged model + processor -> {args.output}")


if __name__ == "__main__":
    main()
