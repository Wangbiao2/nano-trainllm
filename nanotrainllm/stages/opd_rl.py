"""On-policy distillation as a *policy gradient*: the teacher enters the advantage.

`opd.py` makes the divergence the objective.  This file keeps GRPO's objective --
clipped surrogate, group baseline, rule rewards -- and only changes what a token's
advantage is:

    A_t = A_group + teacher_kl_coef * (log p_T(y_t) - log p_S(y_t))

`A_group` is the whitened group reward, constant along the completion; the bracket is
the signed k1 estimator of the reverse KL *at token t*, on the token actually sampled.
One update then carries two kinds of credit: a sequence-level one that knows whether
the answer was right, and a token-level one that knows which tokens the teacher would
have preferred.  A scalar reward cannot express the second, a pure divergence loss
cannot express the first.

**The teacher term is a reward, so it does not backpropagate.**  `logp` is detached
first; differentiating through it would add a `-grad log p_S` term and turn the
objective into neither GRPO nor distillation.

**k1 in the advantage, k3 in the log.**  The signed difference says which direction to
move; `exp(d) - d - 1` is non-negative, so as a reward it could only say "avoid
disagreeing".  It is the right thing to *watch*: `teacher_kl` rising while the rule
reward also rises is the signature of reward hacking.

It costs one extra forward per micro-batch -- the teacher needs only its log-prob of
the sampled tokens, not a whole distribution as in `opd.py`.
"""

from __future__ import annotations

import torch

from ..rl.advantage import expand_to_per_token, teacher_kl
from ..rl.rollout import completion_mask
from .grpo import GRPO, sequence_logprobs


class OnPolicyDistillRL(GRPO):
    def __init__(self, cfg, policy, ref=None, teacher=None, rollout=None):
        super().__init__(cfg, policy, ref, teacher, rollout)
        assert teacher is not None, "--stage opd-rl needs a teacher"
        # Both models score the *same* token ids, so the ids must mean the same
        # thing; hidden sizes may differ freely.
        assert policy.lm_head.out_features == teacher.lm_head.out_features, (
            "student and teacher vocabularies differ; log p_T(y_t) would be the "
            "log-prob of a different token"
        )
        self._teacher_kl = 0.0

    def advantages(self, batch, mask, adv, logp) -> torch.Tensor:
        """GRPO's constant, plus the per-token teacher log-ratio.

        The teacher is scored on `completion_mask` rather than on `mask`:
        `--overlong-filter` can empty `mask` for a whole micro-batch and the gather
        then has no rows to concatenate.  Weighting still happens through `mask`.
        """
        cfg = self.cfg
        scored = completion_mask(batch, self.policy.device)
        with torch.no_grad():
            t_logp = sequence_logprobs(self.teacher, batch, scored, cfg)
        student = logp.detach()
        self._teacher_kl = float(teacher_kl(t_logp, student, mask).sum())
        return expand_to_per_token(adv, mask, t_logp, student, cfg.teacher_kl_coef)

    def loss(self, batch, denom: float) -> tuple[torch.Tensor, dict]:
        """GRPO's loss, with the monitoring KL attached.

        `_teacher_kl` is *overwritten* by `advantages` on every call, not
        accumulated: a running total on `self` would be summed once per micro-batch
        and come out quadratic in `grad_accum`.
        """
        loss, metrics = super().loss(batch, denom)
        metrics["teacher_kl"] = self._teacher_kl
        return loss, metrics
