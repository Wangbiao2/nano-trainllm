"""Prompts in, completions out.  Two backends behind one 3-line protocol.

All the rest of the framework wants to know is which tokens were generated, what they
decode to, and whether generation stopped on its own -- that is `Completion`.

**`HFRollout` is the default, and that is a correctness choice.**  It calls
`model.generate` on the very weights that are about to be updated: no weight sync to get
wrong, no second KV-cache implementation to disagree with the trainer's numerics, and a
breakpoint in the sampler lands in the same process as one in the loss.  Keeping a
separate engine in sync is most of what `VLLMRollout` below has to say.

**Each prompt is generated on its own**, tiled `G` times into the batch.  Batching
different prompts would need left padding, and left padding breaks M-RoPE, which counts
positions from the start of the row.  Tiling one prompt means every row has identical
length, so no padding exists at all -- at the cost of one `generate` call per prompt and
`G` runs of the vision tower on the same image.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from ..template import IGNORE, STOP_TOKEN_IDS, Encoded


@dataclass(slots=True)
class Completion:
    """One sampled continuation of one prompt."""

    prompt: Encoded
    token_ids: list[int]  # generated only; includes the stop token when there is one
    text: str  # decoded, special tokens stripped
    truncated: bool  # True => hit max_completion_length without stopping
    group: int  # index of the prompt within the rollout batch

    def __len__(self) -> int:
        return len(self.token_ids)


def to_encoded(c: Completion) -> Encoded:
    """A `Completion` as a trainable row: prompt masked out, completion supervised.

    The vision tensors come straight from the prompt, so the image is encoded once per
    *prompt* and shared by all `G` rows of its group.
    """
    p = c.prompt
    ids = p.input_ids + c.token_ids
    return Encoded(
        input_ids=ids,
        labels=[IGNORE] * len(p.input_ids) + list(c.token_ids),
        mm_token_type_ids=p.mm_token_type_ids + [0] * len(c.token_ids),
        pixel_values=p.pixel_values,
        image_grid_thw=p.image_grid_thw,
        prompt_len=len(p.input_ids),
        meta=dict(p.meta, truncated=c.truncated, group=c.group, text=c.text),
    )


def completion_mask(batch, device=None) -> torch.Tensor:
    """`[B, T]` 1 on generated tokens, aligned to the *shifted* frame.

    The RL loss lives on `logp[i] = log p(input_ids[i+1] | ...)`, so column `i` here
    refers to `labels[i + 1]`.  Deriving it from `labels` rather than from `prompt_lens`
    makes it right in the padding-free layout too, where one row holds several
    sub-sequences.
    """
    m = (batch.labels[:, 1:] != IGNORE).to(torch.float32)
    return m if device is None else m.to(device)


def collapse_image_runs(ids: list[int], image_token_id: int) -> list[int]:
    """Each run of consecutive `<|image_pad|>` tokens -> a single one.

    vLLM applies its own prompt replacement even to a prompt handed over as token ids, so
    giving it our already-expanded `N` placeholders yields `2N - 1` of them: a prompt that
    no longer matches the one the trainer scores, and no error anywhere.  Collapsing first
    makes vLLM re-expand to exactly `N`, which `VLLMRollout.generate` then asserts.
    """
    out: list[int] = []
    for t in ids:
        if t == image_token_id and out and out[-1] == image_token_id:
            continue
        out.append(t)
    return out


class Rollout:
    """The whole protocol: generate, and (for engines that need it) sync weights."""

    def __init__(self, policy, tokenizer, cfg):
        self.policy = policy
        self.tokenizer = tokenizer
        self.cfg = cfg

    def generate(self, prompts: list[Encoded], n: int) -> list[Completion]:
        raise NotImplementedError

    def sync_weights(self) -> None:
        """No-op unless the sampler holds a second copy of the parameters."""

    def _trim(self, prompt: Encoded, tokens: list[int], group: int) -> Completion:
        """Cut at the first stop token, keeping it.

        Keeping it is not cosmetic: the stop token is a sampled action like any other, and
        excluded from the mask the policy gradient never rewards stopping, so completions
        get longer every step.  Rows that hit the length cap have no stop token and are
        marked `truncated` -- their reward is not a fair sample of the policy, which is
        what `--overlong-filter` is for.
        """
        cut = len(tokens)
        truncated = True
        for i, t in enumerate(tokens):
            if t in STOP_TOKEN_IDS:
                cut, truncated = i + 1, False
                break
        tokens = tokens[:cut]
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return Completion(prompt, tokens, text, truncated, group)


class HFRollout(Rollout):
    def _gen_kwargs(self) -> dict:
        """`use_cache=True` on purpose, even though `build_model` set it False.

        The training forward must not build a KV cache (double the activation memory for
        nothing), but decoding without one is quadratic.  The sampling knobs are only
        passed when we are actually sampling: `temperature` alongside `do_sample=False` is
        a contradiction `GenerationConfig` warns about rather than resolves.
        """
        cfg = self.cfg
        kw = dict(
            max_new_tokens=cfg.max_completion_length,
            use_cache=True,
            eos_token_id=list(STOP_TOKEN_IDS),
            pad_token_id=cfg.pad_token_id,
        )
        if cfg.temperature <= 0.0:
            kw["do_sample"] = False
            return kw
        kw.update(do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p)
        if cfg.top_k > 0:
            kw["top_k"] = cfg.top_k
        return kw

    @torch.no_grad()
    def generate(self, prompts: list[Encoded], n: int) -> list[Completion]:
        """-> `len(prompts) * n` completions, group-major.

        Group-major (`prompt i, sample j` at `i * n + j`) is the layout `rl/advantage.py`
        reshapes to `[B, G]`, so this ordering is load-bearing.  `eval()` matters for more
        than dropout: HF disables gradient checkpointing when `self.training` is False, and
        checkpointing would otherwise force `use_cache=False` and make every decode step
        recompute the prefix.
        """
        model = self.policy.model
        was_training = model.training
        model.eval()
        out: list[Completion] = []
        try:
            with self.policy.gathered():
                for gi, p in enumerate(prompts):
                    out += self._one(model, p, gi, n)
        finally:
            model.train(was_training)
            # generate() writes model.model.rope_deltas; the next training forward
            # must not read it.  See model.py's header, point 2.
            self.policy.reset_rope_cache()
        return out

    def _one(self, model, prompt: Encoded, group: int, n: int) -> list[Completion]:
        dev = self.policy.device
        ids = torch.tensor(prompt.input_ids, device=dev).unsqueeze(0).repeat(n, 1)
        kw = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "mm_token_type_ids": torch.tensor(prompt.mm_token_type_ids, device=dev)
            .unsqueeze(0)
            .repeat(n, 1),
        }
        if prompt.pixel_values is not None:
            kw["pixel_values"] = prompt.pixel_values.to(dev, self.policy.dtype).repeat(n, 1)
            kw["image_grid_thw"] = prompt.image_grid_thw.to(dev).repeat(n, 1)
        seqs = model.generate(**kw, **self._gen_kwargs())
        return [
            self._trim(prompt, seqs[j, ids.shape[1] :].tolist(), group) for j in range(n)
        ]


class VLLMRollout(Rollout):
    """The same protocol, backed by an in-process vLLM engine.

    Five decisions, each of which is a bug if made the other way:

    1. **The engine is built lazily, in this process.**  Lazily so a `VLLMRollout` can be
       constructed on a CPU-only box; in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`)
       because syncing to a separate engine process means serialising 4.9B parameters
       through a pipe every step.
    2. **The adapter is folded in before the copy.**  `load_weights` takes one tensor per
       checkpoint name and has never heard of `lora_A`, so the sync is `merged_adapter()`
       + `hf_named_parameters()` on one card, or `merged_full_state()` under FSDP2.
    3. **`generate` syncs first, every time.**  A backend whose correctness depends on the
       caller remembering a second method goes off-policy after step 1.
    4. **Sleep between rollouts.**  Colocated with the trainer, the engine's weights and KV
       cache compete for the same card as the optimiser state; `sleep(1)` offloads them,
       and the sync after `wake_up()` overwrites them anyway.
    5. **Under FSDP2 it is one tensor-parallel engine across the ranks.**  vLLM's
       `external_launcher` backend spawns no workers and reuses the process group torchrun
       already built.  It is SPMD: every rank must call `generate` with identical inputs,
       so `generate` all-gathers the per-rank prompt shards and slices this rank's
       completions back out afterwards.
    """

    def __init__(self, policy, tokenizer, cfg):
        super().__init__(policy, tokenizer, cfg)
        # On one card vLLM can only sync merged LoRA (full-parameter tuning needs
        # FSDP2, whose shards this path cannot gather); under FSDP2 both work, because
        # `merged_full_state` all-gathers each DTensor shard before handing it over.
        assert cfg.tuner == "lora" or cfg.fsdp2, (
            "the single-process vllm backend syncs merged LoRA weights; use "
            "--tuner lora, or --fsdp2 to sync full-parameter weights across ranks"
        )
        self._llm = None

    # -- the engine --------------------------------------------------------
    @property
    def llm(self):
        """The engine, built on first use and then kept asleep between rollouts."""
        if self._llm is None:
            self._llm = self._build()
        return self._llm

    def _build(self):
        """`enforce_eager=True`: capturing CUDA graphs costs minutes of startup and a
        GiB per shape, which a rollout sharing the card never earns back.  Under FSDP2
        the engine is tensor-parallel across the ranks and loaded with dummy weights --
        the first `generate` syncs the real ones in anyway."""
        cfg = self.cfg
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        from vllm import LLM

        kw = dict(
            dtype="bfloat16",
            seed=cfg.seed,
            gpu_memory_utilization=cfg.vllm_gpu_util,
            max_model_len=self._max_model_len(),
            enforce_eager=True,
            enable_sleep_mode=cfg.vllm_sleep,
            # Disable vLLM's multimodal preprocessor cache.  A group submits `n` child
            # requests sharing one image; the sender-side cache marks the duplicates as
            # already sent while the bounded receiver LRU evicts the item before the
            # last child is served, and the lookup then asserts on the missing hash.
            # The cache only memoises preprocessing, so this changes no output.
            mm_processor_cache_gb=0,
        )
        if cfg.fsdp2:
            kw.update(
                distributed_executor_backend="external_launcher",
                tensor_parallel_size=cfg.world_size,
                load_format="dummy",
            )
        llm = LLM(cfg.model, **kw)
        if cfg.vllm_sleep:
            llm.sleep(1)
        return llm

    def _max_model_len(self) -> int:
        cfg = self.cfg
        if cfg.vllm_max_model_len > 0:
            return cfg.vllm_max_model_len
        return cfg.max_length + cfg.max_completion_length

    def _sampling_params(self, n: int):
        """Our knobs map onto vLLM's with no translation, which is not luck.

        `detokenize=False` because `_trim` decodes: the text must come from the same
        tokenizer call for both backends, and vLLM would otherwise detokenise the
        *untrimmed* completion.

        The one genuine disagreement is greedy sampling of a group: HF returns `n` identical
        rows, vLLM raises.  vLLM is right -- a group with no variance has a zero advantage.
        """
        cfg = self.cfg
        from vllm import SamplingParams

        assert n == 1 or cfg.temperature > 0.0, (
            f"greedy sampling would return {n} identical completions, and a group "
            "with no variance has an identically zero advantage; raise --temperature"
        )
        return SamplingParams(
            n=n,
            temperature=max(cfg.temperature, 0.0),
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            max_tokens=cfg.max_completion_length,
            stop_token_ids=list(STOP_TOKEN_IDS),
            detokenize=False,
        )

    def _prompt(self, p: Encoded):
        """One `Encoded` -> one vLLM prompt, with the placeholder runs collapsed."""
        from vllm import TokensPrompt

        ids = collapse_image_runs(p.input_ids, self.cfg.image_token_id)
        if not p.images:
            return TokensPrompt(prompt_token_ids=ids)
        return TokensPrompt(prompt_token_ids=ids, multi_modal_data={"image": list(p.images)})

    # -- the protocol ------------------------------------------------------
    def generate(self, prompts: list[Encoded], n: int) -> list[Completion]:
        """-> `len(prompts) * n` completions, group-major, same as `HFRollout`.

        One call for all prompts: unlike the HF path there is no padding question, because
        vLLM batches at the token level and prefix-caches a group's shared prompt for free.
        Under FSDP2 all ranks must submit the *same* prompts (SPMD), hence the gather and
        the slice back to this rank's shard.

        The prompt-id assert is the whole reason `collapse_image_runs` exists: if vLLM's
        prompt updates fired again, every log-probability would be computed at a shifted
        position and nothing would raise.
        """
        cfg = self.cfg
        submit, offset = prompts, 0
        if cfg.fsdp2 and cfg.world_size > 1:
            submit, offset = self._gather_prompts(prompts)
        llm = self.llm
        if cfg.vllm_sleep:
            llm.wake_up()
        try:
            self.sync_weights()
            outs = llm.generate(
                [self._prompt(p) for p in submit],
                self._sampling_params(n),
                use_tqdm=(cfg.rank == 0),
            )
        finally:
            if cfg.vllm_sleep:
                llm.sleep(1)
        outs = outs[offset : offset + len(prompts)]
        res: list[Completion] = []
        for gi, (p, out) in enumerate(zip(prompts, outs)):
            assert list(out.prompt_token_ids) == p.input_ids, (
                f"vllm re-expanded the prompt: {len(out.prompt_token_ids)} ids in, "
                f"{len(p.input_ids)} expected -- see collapse_image_runs"
            )
            assert len(out.outputs) == n
            res += [self._trim(p, list(o.token_ids), gi) for o in out.outputs]
        return res

    def _gather_prompts(self, prompts: list[Encoded]) -> tuple[list[Encoded], int]:
        """All-gather every rank's prompt shard into one global batch.

        Returns `(global_prompts, offset)` where `offset` is where this rank's own prompts
        start.  `all_gather_object` because `Encoded` carries token ids and PIL images, not
        a fixed-shape tensor.
        """
        import torch.distributed as dist

        gathered: list = [None] * self.cfg.world_size
        dist.all_gather_object(gathered, prompts)
        offset = sum(len(gathered[r]) for r in range(self.cfg.rank))
        return [p for shard in gathered for p in shard], offset

    def sync_weights(self) -> None:
        """Fold the adapter in, copy every parameter across, drop the prefix cache.

        On one card the merge is peft's own, in place.  Under FSDP2 the parameters are
        DTensor shards peft cannot merge in place, so `merged_full_state()` folds on the
        all-gathered tensor instead.  `reset_prefix_cache()` is not optional: cached blocks
        are keyed by token ids alone, so after an update they would serve KV computed by the
        *old* policy.
        """
        if self.cfg.fsdp2:
            from ..model import merged_full_state

            items = list(merged_full_state(self.policy.model))
            self.llm.apply_model(lambda m: m.load_weights(items))
            self.llm.reset_prefix_cache()
            return

        from ..model import hf_named_parameters

        with self.policy.merged_adapter():
            items = list(hf_named_parameters(self.policy.model))
            self.llm.apply_model(lambda m: m.load_weights(items))
        self.llm.reset_prefix_cache()


def build_rollout(cfg, policy, tokenizer) -> Rollout:
    if cfg.rollout == "vllm":
        return VLLMRollout(policy, tokenizer, cfg)
    assert cfg.rollout == "hf", f"unsupported rollout backend {cfg.rollout!r}"
    return HFRollout(policy, tokenizer, cfg)
