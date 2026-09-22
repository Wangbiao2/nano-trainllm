"""The stage protocol.  Three questions, and nothing else.

Every stage -- pt, sft, dpo, grpo, opd, opd-rl -- answers the same three questions,
and `loop.py` asks them in the same order regardless of which is running:

1. **`micro_batches(units)`** -- what forward passes does one optimiser step consist
   of?  Off-policy stages just collate what the dataset handed them; GRPO turns
   prompts into `G` completions each.  Rollout living here rather than in the loop is
   what keeps `loop.py` free of `if stage == "grpo"`.

2. **`denom(batches)`** -- what does the summed loss get divided by?  Computed over the
   *whole* step and reduced across ranks *before* any backward, because that is the
   only way `micro_batch=1 x accum=4` and `micro_batch=4 x accum=1` produce the same
   gradient.  Dividing each micro-batch by its own token count is the classic silent bug.

3. **`loss(batch, denom)`** -- the scalar to call `.backward()` on, plus a dict of
   already-summed scalars to log.  `metrics` values are summed across micro-batches by
   the loop, so they must be extensive (a token count, a summed NLL), not means.
"""

from __future__ import annotations

import torch

from ..collate import collate


def all_sum(value: float, device=None) -> float:
    """Sum a python scalar over ranks.  Identity when not running distributed."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return float(value)
    t = torch.tensor([float(value)], dtype=torch.float64, device=device)
    dist.all_reduce(t)
    return float(t)


class Stage:
    """Base class holding the collaborators a stage may need.

    `ref`, `teacher` and `rollout` are `None` unless the stage asked for them --
    `cli.py` decides that from `cfg.needs_ref_model` / `cfg.needs_rollout`, so a
    plain SFT run never pays for a second copy of the weights.
    """

    def __init__(self, cfg, policy, ref=None, teacher=None, rollout=None):
        self.cfg = cfg
        self.policy = policy
        self.ref = ref
        self.teacher = teacher
        self.rollout = rollout

    def micro_batches(self, units: list[list]) -> list:
        return [collate(u, self.cfg) for u in units]

    def denom(self, batches: list) -> float:
        raise NotImplementedError

    def loss(self, batch, denom: float) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError


def build(cfg, **kw) -> Stage:
    """`cfg.stage` -> the one object `loop.py` talks to.

    Imports are local so that a `--stage sft` run never imports the RL code, and
    so that a broken RL file cannot break an SFT run.
    """
    if cfg.stage in ("pt", "sft"):
        from .sft import CrossEntropy

        return CrossEntropy(cfg, **kw)
    if cfg.stage == "dpo":
        from .dpo import DPO

        return DPO(cfg, **kw)
    if cfg.stage == "opd":
        from .opd import OnPolicyDistill

        return OnPolicyDistill(cfg, **kw)
    if cfg.stage == "grpo":
        from .grpo import GRPO

        return GRPO(cfg, **kw)
    if cfg.stage == "opd-rl":
        from .opd_rl import OnPolicyDistillRL

        return OnPolicyDistillRL(cfg, **kw)
    raise AssertionError(f"no stage implementation for {cfg.stage!r}")
