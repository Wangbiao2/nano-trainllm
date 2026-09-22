"""Load the model, decide what trains, and hand back hidden states -- never logits.

Four things that are easy to get wrong:

1. **Hidden states, not logits.**  `Policy.hidden` stops at `model.model` and
   `logprobs.py` applies `lm_head` in chunks; the whole-model forward would
   materialise `[B, S, ~250k]` -- 4 GiB per 8k sequence, the fastest way to OOM.
2. **`rope_deltas` is mutable state on the model.**  `generate()` writes it and a later
   forward reads it when `position_ids is None`.  We always pass them, and a GRPO loop
   still calls `reset_rope_cache()` after every rollout.
3. **LoRA is injected, not wrapped.**  `inject_adapter_in_model` returns the *same*
   object, so `model.model` and `model.lm_head` keep their meaning; `get_peft_model`
   would bury both under `.base_model.model`.  The price is freezing the base weights
   ourselves.
4. **Padding-free needs a varlen kernel.**  Under sdpa the `full_attention` layers --
   every 4th -- would attend across sub-sequence boundaries and never complain, so
   `build_model` asserts the combination away.

`tiny_model` keeps the real vocabulary and token ids and shrinks only depth and width,
so a real encoded batch -- image included -- flows through it on CPU.  That is what
makes `--dry-run` GPU-free.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn

# One full_attention layer at index 3 plus three linear_attention ones: both code
# paths are exercised, which is the point of keeping four layers rather than two.
TINY_TEXT = {
    "num_hidden_layers": 4,
    "layer_types": ["linear_attention"] * 3 + ["full_attention"],
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "linear_key_head_dim": 32,
    "linear_value_head_dim": 32,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
}
TINY_VISION = {
    "depth": 2,
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_heads": 4,
    "out_hidden_size": 128,
}


def shrink(hf_config):
    """A tiny random model with the *real* vocabulary and the real token ids.

    Only depth and width change.  `mrope_section` must keep summing to
    `head_dim * partial_rotary_factor / 2` -- that is how many rotary pairs the three
    M-RoPE axes are split across, and getting it wrong is an obscure shape error.
    """
    import copy

    cfg = copy.deepcopy(hf_config)
    for k, v in TINY_TEXT.items():
        setattr(cfg.text_config, k, v)
    for k, v in TINY_VISION.items():
        setattr(cfg.vision_config, k, v)
    rope = cfg.text_config.rope_parameters
    half = int(TINY_TEXT["head_dim"] * rope["partial_rotary_factor"] / 2)
    rope["mrope_section"] = [half - 2, 1, 1]
    assert sum(rope["mrope_section"]) == half
    return cfg


def build_model(cfg, adapters: bool = True) -> nn.Module:
    """-> `Qwen3_5ForConditionalGeneration`, frozen and LoRA-injected per `cfg`.

    `adapters=False` is the teacher/reference escape hatch.  Asking for that by flipping
    `cfg.tuner` to `"full"` would trip the "full-parameter tuning needs --fsdp2" assert,
    which is about optimiser state neither of them allocates.
    """
    from transformers import AutoConfig, AutoModelForImageTextToText

    hf_config = cfg.hf_config or AutoConfig.from_pretrained(cfg.model)
    mt = hf_config.model_type
    assert mt == "qwen3_5", f"only qwen3_5 is supported, got {mt}"
    if cfg.padding_free:
        assert cfg.attn_impl.startswith("flash_attention"), (
            "padding-free needs a varlen attention kernel; sdpa/eager would let the "
            "full_attention layers see across sub-sequence boundaries silently"
        )

    if cfg.tiny_model:
        model = AutoModelForImageTextToText.from_config(
            shrink(hf_config), dtype=torch.float32, attn_implementation=cfg.attn_impl
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            cfg.model, dtype=torch.bfloat16, attn_implementation=cfg.attn_impl
        )
    model.config.use_cache = False
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable({"use_reentrant": False})
    freeze(model, cfg)
    if adapters and cfg.tuner == "lora":
        inject_lora(model, cfg)
    return model


def freeze(model: nn.Module, cfg) -> None:
    """Vision tower and/or its merger off, by name.

    `visual.merger` is the aligner -- the MLP that maps ViT features into the text
    hidden size.  Frozen tower + trained merger is the usual VL recipe, hence two
    flags rather than one.
    """
    visual = model.model.visual
    if cfg.freeze_vit:
        visual.requires_grad_(False)
    visual.merger.requires_grad_(not cfg.freeze_aligner)


def inject_lora(model: nn.Module, cfg) -> None:
    """Freeze everything, add LoRA to the text side, unfreeze only `lora_`.

    `inject_adapter_in_model` does not freeze the base weights for you -- that is
    `get_peft_model`'s job -- hence the explicit loop, which is also the clearest
    possible statement of what LoRA trains.
    """
    from peft import LoraConfig, inject_adapter_in_model

    model.requires_grad_(False)
    inject_adapter_in_model(
        LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        ),
        model,
    )
    n = 0
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad_(True)
            n += p.numel()
    assert n > 0, f"no LoRA parameters matched {cfg.lora_target_modules}"


def hf_named_parameters(model: nn.Module):
    """`(name, tensor)` pairs under the names the *checkpoint* uses.

    LoRA injection renames things: a targeted `q_proj` keeps its matrix at
    `q_proj.base_layer.weight` with the two factors beside it.  An inference engine has
    never heard of either spelling, so drop the factors and undo `.base_layer.`.  Only
    meaningful inside `Policy.merged_adapter()`, where the base matrices already contain
    the adapter's contribution -- outside it the values are the un-finetuned weights.
    """
    for name, p in model.named_parameters():
        if "lora_" in name:
            continue
        yield name.replace(".base_layer.", "."), p.detach()


def merged_full_state(model: nn.Module):
    """`(checkpoint_name, full_tensor)` for an inference copy, gathered under FSDP2.

    The vLLM sync under `--fsdp2` cannot go through `Policy.merged_adapter()`: peft's
    `merge()` does `W += scaling * B @ A` in place, and under `fully_shard` all three are
    DTensor shards with mismatched placements.  So the fold happens here, on plain
    tensors, *after* each one is all-gathered -- one at a time, so the model is never
    resident unsharded all at once.  `full_tensor()` is a collective: every rank must
    iterate this generator in the same order and to the end, which `load_weights` does.
    """
    from torch.distributed.tensor import DTensor

    # base-weight checkpoint name -> (lora_A, lora_B, scaling) for each adapted linear
    deltas: dict = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
            deltas[f"{name}.base_layer.weight"] = (
                mod.lora_A["default"].weight,
                mod.lora_B["default"].weight,
                mod.scaling["default"],
            )

    def full(t: torch.Tensor) -> torch.Tensor:
        return t.full_tensor() if isinstance(t, DTensor) else t

    for name, p in model.named_parameters():
        if "lora_" in name:
            continue
        w = full(p.detach())
        if name in deltas:
            a, b, scaling = deltas[name]
            w = w + (scaling * (full(b.detach()) @ full(a.detach()))).to(w.dtype)
        yield name.replace(".base_layer.", "."), w


def shard_fsdp2(model: nn.Module, cfg) -> None:
    """`fully_shard` every decoder layer and vision block, then the root.

    Innermost-first is required: `fully_shard` on the root only shards what it still
    owns directly.  `lm_head` is tied to `embed_tokens` and stays with the root --
    sharding it separately would give one storage two placements.
    """
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for block in model.model.visual.blocks:
        fully_shard(block, mp_policy=policy)
    for layer in model.model.language_model.layers:
        fully_shard(layer, mp_policy=policy)
    fully_shard(model, mp_policy=policy)

    # `Policy.hidden` calls `model.model(...)` and `logprobs.py` applies `lm_head` by
    # hand, so the root's own forward -- and its pre-forward all-gather -- never runs:
    # `embed_tokens` and the tied `lm_head` stay sharded and the first embedding lookup
    # dies with "mixed Tensor and DTensor".  Registering the base-model forward as an
    # FSDP forward method gathers them there and installs the hooks that reduce-scatter
    # their gradients, which full-parameter tuning needs.

    import types

    from torch.distributed.fsdp import register_fsdp_forward_method

    def _hidden_forward(m, **kwargs):
        return m.model(**kwargs)

    model._hidden_forward = types.MethodType(_hidden_forward, model)
    register_fsdp_forward_method(model, "_hidden_forward")


class Policy:
    """A model plus the two calls the rest of the framework is allowed to make.

    Deliberately not an `nn.Module`: it holds no parameters, and making it one would
    invite `Policy(...)` calls that return logits.
    """

    def __init__(self, model: nn.Module, cfg):
        self.model = model
        self.cfg = cfg

    @property
    def lm_head(self) -> nn.Module:
        return self.model.lm_head

    @property
    def dtype(self) -> torch.dtype:
        return self.model.lm_head.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.model.lm_head.weight.device

    def hidden(self, batch) -> torch.Tensor:
        """`[B, S, H]` -- the last hidden state, with no LM head applied.

        Under FSDP2 this must go through `_hidden_forward` (see `shard_fsdp2`);
        calling `self.model.model` directly would hit a sharded-DTensor embedding.
        """
        kwargs = batch.to(self.device, self.dtype).model_kwargs()
        forward = getattr(self.model, "_hidden_forward", None)
        out = forward(**kwargs) if forward is not None else self.model.model(**kwargs)
        return out.last_hidden_state

    def reset_rope_cache(self) -> None:
        self.model.model.rope_deltas = None

    @contextmanager
    def gathered(self):
        """Unshard the FSDP2 model for a rollout, then put it back.

        Every FSDP2 forward all-gathers, and each rank decodes a different prompt for a
        different number of steps, so collectives inside `generate` deadlock.  Gathering
        once -- with `reshard_after_forward` off so the loop never re-gathers -- removes
        every collective from generation.  No-op without FSDP2, so `HFRollout` can wrap
        every `generate` in it unconditionally.

        Full parameters then sit on every rank, ~10 GB for 4.9B in bf16.  It fits only
        because rollout is inference: no optimiser state, no activation graph.  A model
        too large for one card un-sharded needs the vLLM backend instead.
        """
        from torch.distributed.fsdp import FSDPModule

        mods = [m for m in self.model.modules() if isinstance(m, FSDPModule)]
        if not mods:
            yield self
            return
        for m in mods:
            m.set_reshard_after_forward(False, recurse=False)
            m.unshard()
        try:
            yield self
        finally:
            for m in mods:
                m.reshard()
                # Restore fully_shard's defaults, which `shard_fsdp2` relies on: the
                # root keeps its parameters after forward (backward needs them at
                # once), everything else reshards to give the memory back.
                m.set_reshard_after_forward(m is not self.model, recurse=False)

    @contextmanager
    def no_adapter(self):
        """Run the base model: the LoRA deltas switched off, then back on.

        peft computes `W x + (alpha / r) * B A x`, so disabling every adapter layer
        recovers the frozen base model *exactly* -- a LoRA policy is its own reference
        model, which is why `stages/dpo.py` needs no second copy of the weights.
        Nothing else has to be turned off: `lora_dropout` lives inside the layers this
        disables and `attention_dropout` is 0.0, so the reference log-probs are
        reproducible even in train mode.
        """
        from peft.tuners.tuners_utils import BaseTunerLayer

        layers = [m for m in self.model.modules() if isinstance(m, BaseTunerLayer)]
        assert layers, "no adapters to disable; a full-parameter run needs a real ref model"
        for m in layers:
            m.enable_adapters(False)
        try:
            yield self
        finally:
            for m in layers:
                m.enable_adapters(True)

    @contextmanager
    def merged_adapter(self):
        """The inverse of `no_adapter()`: fold `B A` into `W`, then unfold it.

        vLLM's `load_weights` wants one tensor per weight, not a base matrix plus two
        factors.  **The round trip is not bit-exact in bf16** -- `(W + D) - D` loses
        whatever fell off the mantissa -- so it is scoped to one rollout and undone
        immediately.  A standing merge would also make `no_adapter()`, and therefore
        DPO's reference model, silently wrong.
        """
        from peft.tuners.tuners_utils import BaseTunerLayer

        layers = [m for m in self.model.modules() if isinstance(m, BaseTunerLayer)]
        assert layers, "no adapters to merge; a full-parameter run has nothing to fold"
        for m in layers:
            m.merge()
        try:
            yield self
        finally:
            for m in layers:
                m.unmerge()

    def trainable(self) -> list[nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def state_dict(self) -> dict:
        """Only what was trained: the LoRA deltas, or everything under `full`.

        Under FSDP2 each value is a sharded DTensor, so saving `model.state_dict()`
        as-is writes one rank's shard -- a half-sized, unloadable checkpoint.
        `full_tensor()` is a collective, so every rank must call this and reach the
        same keys in the same order (registration order, identical across ranks).
        Filtering to the LoRA keys *first* means a lora run gathers a few MB.
        """
        from torch.distributed.tensor import DTensor

        sd = self.model.state_dict()
        if self.cfg.tuner == "lora":
            sd = {k: v for k, v in sd.items() if "lora_" in k}
        return {
            k: (v.full_tensor() if isinstance(v, DTensor) else v).detach().cpu()
            for k, v in sd.items()
        }


def load_policy(cfg, device: str = "cpu") -> Policy:
    model = build_model(cfg).to(device)
    model.train()
    return Policy(model, cfg)


def load_ref(cfg, device: str = "cpu") -> Policy:
    """`pi_ref` as a second frozen copy of the base checkpoint: no adapters, eval.

    Only `stages/grpo.py` needs this, and only when `--kl-coef` is non-zero; DPO gets
    the identical distribution for free from `Policy.no_adapter()` under LoRA, and a
    second 4.9B copy is 10 GB of a model we already have.
    """
    from dataclasses import replace

    rcfg = replace(cfg, gradient_checkpointing=False)
    model = build_model(rcfg, adapters=False).to(device)
    model.requires_grad_(False)
    model.eval()
    return Policy(model, rcfg)


def load_teacher(cfg, device: str = "cpu") -> Policy:
    """The OPD teacher: same architecture, frozen, eval mode, no adapters.

    In-process rather than behind an HTTP endpoint precisely so that
    `union_topk_logprobs` can score the teacher at the *student's* indices -- a remote
    top-k API cannot answer that question.
    """
    from dataclasses import replace

    assert cfg.teacher
    tcfg = replace(cfg, model=cfg.teacher, gradient_checkpointing=False)
    model = build_model(tcfg, adapters=False).to(device)
    model.requires_grad_(False)
    model.eval()
    return Policy(model, tcfg)
