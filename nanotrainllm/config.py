"""The one and only config object.

ONE flat dataclass: no config-of-configs, no yaml, no OmegaConf.  Every knob six
stages have is in this file, with one inline note each.

  * A knob exists only if something exercises it.  Variants that merely name an
    alternative from a paper were left out; the docstrings explain the families.
  * `__post_init__` validates with bare `assert`, which doubles as documentation
    of what is *not* supported.  It must stay idempotent -- `dataclasses.replace`
    re-runs it when the ref and teacher configs are derived.
  * Sentinel fields (`-1` / `None`) are filled in later by other components,
    which is why this dataclass is not frozen.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any

STAGES = ("pt", "sft", "dpo", "grpo", "opd", "opd-rl")

# Stages that need a rollout (i.e. the model generates its own training data).
ON_POLICY_STAGES = ("grpo", "opd-rl")

# Stages whose dataset rows carry a (chosen, rejected) pair instead of one target.
PAIRWISE_STAGES = ("dpo",)


@dataclass(slots=True)
class Config:
    # ---- what to run ----------------------------------------------------
    stage: str = "sft"
    model: str = "Qwen/Qwen3.5-4B"  # a local checkpoint directory, or a hub repo id
    teacher: str | None = None  # opd / opd-rl only; loaded in-process
    data: str = ""  # jsonl path
    output_dir: str = "output"
    seed: int = 42

    # ---- sequence / batching -------------------------------------------
    max_length: int = 4096
    micro_batch_size: int = 1
    grad_accum: int = 1
    packing: bool = False  # best-fit pack many samples into one max_length row
    padding_free: bool = False  # concat a batch into one row, no pad tokens
    truncation: str = "right"  # right | left | drop
    max_pixels: int = -1  # -1 => processor default; else cap each image's h*w area

    # ---- optimisation ---------------------------------------------------
    epochs: int = 1
    max_steps: int = -1  # -1 => derived from epochs
    lr: float = 1e-5
    lr_min_ratio: float = 0.1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    max_grad_norm: float = 1.0

    # ---- memory / parallelism ------------------------------------------
    tuner: str = "lora"  # lora | full
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    fsdp2: bool = False
    gradient_checkpointing: bool = True
    freeze_vit: bool = True
    freeze_aligner: bool = False
    attn_impl: str = "flash_attention_2"
    logit_chunk: int = 1024  # rows per chunk in logprobs.py; the vocab is ~250k

    # ---- generation (grpo / opd / opd-rl) -------------------------------
    rollout: str = "hf"  # hf | vllm
    num_generations: int = 4  # G, the GRPO group size
    max_completion_length: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    enable_thinking: bool = True  # False => the template closes <think> immediately
    # vLLM lives in *this* process next to the trainer, so it cannot have vLLM's
    # own 0.9 default: the weights, the optimiser state and the KV cache all come
    # out of one card.  `--vllm-sleep` gives the rest back between rollouts.
    vllm_gpu_util: float = 0.3
    vllm_max_model_len: int = -1  # -1 => max_length + max_completion_length
    vllm_sleep: bool = True

    # ---- grpo / opd-rl --------------------------------------------------
    # Three tiers, three units.  `rollout_batch_size` and `global_batch_size` count
    # *prompts* and are *global* (summed across ranks); `micro_batch_size` counts
    # *completions* per forward pass, so a forward micro-batch is that many
    # completions sliced in order and may span prompts and groups.  -1 falls back to
    # effective_batch_size, so setting neither behaves like a plain micro x accum step.
    rollout_batch_size: int = -1  # prompts generated per rollout (the experience buffer)
    global_batch_size: int = -1  # prompts consumed per optimiser step (the mini-batch)
    num_iterations: int = 1  # mu / ppo-epochs: passes over the rollout buffer
    epsilon_low: float = 0.2
    epsilon_high: float = 0.2
    delta: float = -1.0  # dual-clip upper bound; <0 disables
    policy_loss_type: str = "default"  # gspo / gspo-token => sequence ratio; cispo => no dead zone
    loss_agg_mode: str = "token-mean"
    scale_rewards: bool = True  # False => Dr.GRPO (no std division)
    advantage_estimator: str = "grpo"  # rloo => leave-one-out baseline, not the group mean
    kl_coef: float = 0.0  # KL(policy||ref) added to the loss; 0 => no ref model
    reward_funcs: str = "accuracy,format"
    reward_weights: str = ""  # comma separated; empty => all 1.0
    overlong_filter: bool = False  # drop truncated completions from the loss

    # ---- dpo ------------------------------------------------------------
    dpo_beta: float = 0.1
    dpo_reference_free: bool = False  # True => drop the ref term (a.k.a. CPO-ish)
    dpo_sft_weight: float = 0.0  # >0 adds an NLL term on the chosen response (RPO)

    # ---- opd / opd-rl ---------------------------------------------------
    topk: int = 64
    union_topk: bool = True  # False => teacher-only top-k
    with_tail: bool = True  # append a logsumexp("everything else") bucket
    jsd_beta: float = 1.0  # 0 => forward KL, 1 => reverse KL, else JSD
    on_policy_ratio: float = 1.0  # fraction of samples the student generates
    teacher_kl_coef: float = 1.0  # opd-rl only: weight of the per-token teacher term

    # ---- logging / io ---------------------------------------------------
    log_every: int = 1
    log_samples_every: int = 0  # rollouts between rank-0 sample dumps; 0 => off (rl only)
    save_every: int = -1  # -1 => only at the end
    dry_run: bool = False
    tiny_model: bool = False  # random 2-layer model, CPU-only; what --dry-run uses

    # ---- filled in by other components (sentinels) ----------------------
    hf_config: Any = None
    vocab_size: int = -1
    image_token_id: int = -1
    merge_size: int = -1  # spatial_merge_size; collate.py needs it for M-RoPE
    pad_token_id: int = -1
    eos_token_id: int = -1
    world_size: int = 1
    rank: int = 0

    def __post_init__(self):
        assert self.stage in STAGES, f"stage must be one of {STAGES}, got {self.stage!r}"
        assert self.tuner in ("lora", "full")
        assert self.rollout in ("hf", "vllm")
        assert self.truncation in ("right", "left", "drop")
        assert self.loss_agg_mode in (
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
        )
        assert self.policy_loss_type in ("default", "gspo", "gspo-token", "cispo")
        assert self.advantage_estimator in ("grpo", "rloo")
        assert self.micro_batch_size >= 1 and self.grad_accum >= 1
        # Only the world-independent checks: world_size is still the sentinel 1 until
        # `cli.init_distributed` fills it in, so `loop.py` does the divisibility ones.
        assert self.rollout_batch_size == -1 or self.rollout_batch_size >= 1
        assert self.global_batch_size == -1 or self.global_batch_size >= 1
        if self.rollout_batch_size > 0 and self.global_batch_size > 0:
            assert self.rollout_batch_size % self.global_batch_size == 0, (
                f"rollout_batch_size={self.rollout_batch_size} must be a multiple of "
                f"global_batch_size={self.global_batch_size}"
            )
        assert 0.0 <= self.jsd_beta <= 1.0
        assert 0.0 <= self.on_policy_ratio <= 1.0
        assert self.topk >= 1
        assert self.num_generations >= 1
        assert 0.0 < self.vllm_gpu_util < 1.0
        # GRPO needs >=2 samples per group, otherwise the group-normalised
        # advantage is identically zero and nothing is learned.
        if self.stage in ON_POLICY_STAGES:
            assert self.num_generations >= 2, "GRPO-style stages need num_generations >= 2"
        if self.stage in ("opd", "opd-rl"):
            assert self.teacher, f"stage {self.stage} requires --teacher"
        if self.packing:
            # Assignment, not an assert that padding_free is still False:
            # `dataclasses.replace` re-runs __post_init__ when the ref and teacher
            # configs are derived, so this has to be idempotent.
            self.padding_free = True
            # A packed micro-batch is exactly one max_length bin, so this knob has
            # nothing left to mean; scale the step with --grad-accum instead.
            assert self.micro_batch_size == 1, "--packing requires --micro-batch-size 1"
        if self.tuner == "full" and not self.fsdp2:
            # 4.9B params * 10 bytes (bf16 param + fp32 master + 2 fp32 moments)
            # is ~68 GB of optimiser state; it does not fit on one 40 GB card.
            assert self.tiny_model or self.dry_run, "full-parameter tuning needs --fsdp2"
        if not self.dry_run and not self.tiny_model:
            # A hub repo id ("org/name") is allowed too; transformers resolves it.
            assert os.path.isdir(self.model) or "/" in self.model, f"no such model: {self.model!r}"
            assert self.data and os.path.isfile(self.data), f"not a file: {self.data!r}"

    # -- derived, read-only ------------------------------------------------
    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.grad_accum * self.world_size

    @property
    def needs_rollout(self) -> bool:
        return self.stage in ON_POLICY_STAGES or (
            self.stage == "opd" and self.on_policy_ratio > 0.0
        )

    @property
    def is_on_policy(self) -> bool:
        """A stage whose training data comes from its own rollout, and whose update
        follows the three-tier rollout/global/micro schedule in `loop.py`."""
        return self.stage in ON_POLICY_STAGES

    @property
    def rollout_prompts(self) -> int:
        """Prompts generated per rollout (global).  -1 => the single-step effective batch."""
        return self.rollout_batch_size if self.rollout_batch_size > 0 else self.effective_batch_size

    @property
    def global_prompts(self) -> int:
        """Prompts consumed per optimiser step (global).  -1 => the effective batch."""
        return self.global_batch_size if self.global_batch_size > 0 else self.effective_batch_size

    @property
    def updates_per_rollout(self) -> int:
        """How many optimiser steps one rollout buffer is split into (>=1)."""
        return self.rollout_prompts // self.global_prompts

    @property
    def needs_ref_model(self) -> bool:
        if self.stage == "dpo":
            return not self.dpo_reference_free
        return self.kl_coef != 0.0

    @property
    def is_pairwise(self) -> bool:
        return self.stage in PAIRWISE_STAGES

    @property
    def reward_func_names(self) -> list[str]:
        return [x for x in self.reward_funcs.split(",") if x]

    @property
    def reward_weight_values(self) -> list[float]:
        names = self.reward_func_names
        if not self.reward_weights:
            return [1.0] * len(names)
        w = [float(x) for x in self.reward_weights.split(",") if x]
        assert len(w) == len(names), f"{len(w)} weights for {len(names)} reward funcs"
        return w

    @property
    def lora_target_modules(self) -> list[str]:
        return [x for x in self.lora_target.split(",") if x]

    @classmethod
    def from_kwargs(cls, **kwargs) -> "Config":
        """Build a Config, silently dropping keys we do not have a field for, so
        that `cli.py` can hand over a whole argparse namespace as-is."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in kwargs.items() if k in known})

    def bind_model_meta(self, hf_config, tokenizer) -> None:
        """Fill in the sentinel fields once the checkpoint has been inspected."""
        text = getattr(hf_config, "text_config", hf_config)
        self.hf_config = hf_config
        self.vocab_size = text.vocab_size
        self.image_token_id = getattr(hf_config, "image_token_id", -1)
        # `preprocessor_config.json:merge_size` and `vision_config.spatial_merge_size`
        # are the same number by construction; taking it from the config means the
        # collator can hold a Config alone, with no processor.
        self.merge_size = hf_config.vision_config.spatial_merge_size
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
        assert self.vocab_size > 0 and self.eos_token_id is not None
        assert self.merge_size > 0
