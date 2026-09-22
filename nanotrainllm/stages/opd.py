"""On-policy distillation: the KL *is* the loss.

The student generates, the teacher grades every token of what the student
generated, and the divergence between the two next-token distributions is
backpropagated directly.  No reward, no advantage, no clipping: the signal is
dense (one term per token per vocabulary entry, not one scalar per completion)
and exactly zero once the student agrees.

**The support is a union, and it is finite.**  A full-vocabulary KL would need a
`[tokens, ~250k]` fp32 tensor per side.  `logprobs.union_topk_logprobs` evaluates
both distributions on the union of their top-k sets plus one "everything else"
bucket -- a genuine distribution over a partition of the vocabulary, so the KL
from it is a lower bound rather than an approximation of unknown sign.  The union
rather than the teacher's top-k alone is what lets reverse KL see the tokens the
*student* is confident about, which are precisely the ones it should punish.

**On-policy vs off-policy is one coin flip per micro-batch.**  `1.0` replaces every
micro-batch with the student's own samples, `0.0` distils the dataset's answers
as-is (classic sequence-level KD), in between is GKD's `lmbda`.  A student distilled
only on the teacher's text never learns to recover from its own mistakes.
"""

from __future__ import annotations

import math
import random

import torch

from ..collate import collate
from ..logprobs import shift_for_causal_lm, union_topk_logprobs
from ..rl.rollout import to_encoded
from ..template import IGNORE, Encoded
from . import Stage, all_sum


def _kl(target_logp: torch.Tensor, input_logp: torch.Tensor) -> torch.Tensor:
    """`sum_v p_target * (log p_target - log p_input)` over the last axis.

    The `-inf` columns that pad the support to a rectangle must contribute zero, but
    `0 * (-inf - -inf)` is `nan`, so they are masked *before* the arithmetic.
    """
    dead = ~(torch.isfinite(target_logp) & torch.isfinite(input_logp))
    t = target_logp.masked_fill(dead, 0.0)
    i = input_logp.masked_fill(dead, 0.0)
    return (t.exp() * (t - i)).masked_fill(dead, 0.0).sum(-1)


def divergence(
    student_logp: torch.Tensor, teacher_logp: torch.Tensor, beta: float = 1.0
) -> torch.Tensor:
    """`[N, M]` log-probs on a shared support -> `[N]` per-token divergence.

    With `M = beta * T + (1 - beta) * S`:

        D_beta = (1 - beta) * KL(T || M) + beta * KL(S || M)

    `beta = 0` is the forward KL `KL(T || S)` (mass-covering: the student hedges,
    punished for too little mass where the teacher has some), `beta = 1` the
    reverse KL `KL(S || T)` (mode-seeking: punished for mass the teacher does not
    endorse, so it commits), `beta = 0.5` plain Jensen-Shannon.  Reverse is the
    default -- it means "do not say things the teacher would not say".

    `beta` weights the *teacher* inside `M`, which makes the family continuous at
    both endpoints.  Mix the other way round and `beta -> 0` approaches the reverse
    KL while `beta == 0` is the forward one -- a discontinuity that hides which
    divergence a mid-range `--jsd-beta` optimises.
    """
    assert 0.0 <= beta <= 1.0, beta
    if beta == 0.0:
        return _kl(teacher_logp, student_logp)
    if beta == 1.0:
        return _kl(student_logp, teacher_logp)
    mix = torch.logaddexp(teacher_logp + math.log(beta), student_logp + math.log1p(-beta))
    return (1.0 - beta) * _kl(teacher_logp, mix) + beta * _kl(student_logp, mix)


def prompt_only(e: Encoded) -> Encoded:
    """The generation prefix of a supervised row: everything before its answer.

    Slicing at `prompt_len` makes this byte-identical to the prefix the off-policy
    branch trains on, which is what makes the two branches comparable.  Vision
    tokens all live in the prompt, so `pixel_values` survives intact.
    """
    p = e.prompt_len
    assert p > 0, "an opd row needs a prompt; check template.encode_sft"
    return Encoded(
        input_ids=e.input_ids[:p],
        labels=[IGNORE] * p,
        mm_token_type_ids=e.mm_token_type_ids[:p],
        pixel_values=e.pixel_values,
        image_grid_thw=e.image_grid_thw,
        prompt_len=p,
        meta=e.meta,
        images=e.images,
    )


class OnPolicyDistill(Stage):
    def __init__(self, cfg, policy, ref=None, teacher=None, rollout=None):
        super().__init__(cfg, policy, ref, teacher, rollout)
        assert teacher is not None, "--stage opd needs a teacher"
        # The two models are scored on a *shared* set of vocabulary indices, so
        # they must agree on what those indices mean.  Hidden sizes may differ.
        assert policy.lm_head.out_features == teacher.lm_head.out_features, (
            "student and teacher vocabularies differ; the shared support would be "
            "meaningless"
        )
        if cfg.on_policy_ratio > 0.0:
            assert rollout is not None, "--on-policy-ratio > 0 needs a rollout"
        self._rng = random.Random(cfg.seed)

    # -- generate, or don't ------------------------------------------------
    def micro_batches(self, units: list[list]) -> list:
        """One coin flip per micro-batch, which is GKD's `lmbda`.

        Which branch a row came from goes into its `meta`, so the metric stays
        extensive -- a counter on `self` would be summed once per micro-batch.
        """
        cfg, out = self.cfg, []
        for unit in units:
            if self._rng.random() >= cfg.on_policy_ratio:
                for e in unit:
                    e.meta["on_policy"] = False
                out.append(collate(unit, cfg))
                continue
            prompts = [prompt_only(e) for e in unit]
            rows = [to_encoded(c) for c in self.rollout.generate(prompts, 1)]
            for e in rows:
                e.meta["on_policy"] = True
            out.append(collate(rows, cfg))
        return out

    # -- the divisor -------------------------------------------------------
    def denom(self, batches: list) -> float:
        """Supervised tokens in the step -- the same count `sft.py` uses.

        The divergence is per-token, so a token-mean is the only aggregation that
        makes `micro_batch x accum` invariant.
        """
        n = sum(int((b.labels[:, 1:] != IGNORE).sum()) for b in batches)
        return max(all_sum(n, device=self.policy.device), 1.0)

    # -- the loss ----------------------------------------------------------
    def supports(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> `(student_logp, teacher_logp, keep)` on the union support.

        Both forwards see the *same* batch, so the teacher scores exactly the tokens
        the student is trained on.  Its LM head is already no-grad inside
        `union_topk_logprobs`; this adds the body.
        """
        cfg = self.cfg
        h, y = shift_for_causal_lm(self.policy.hidden(batch), batch.labels)
        keep = y != IGNORE
        with torch.no_grad():
            ht, _ = shift_for_causal_lm(self.teacher.hidden(batch), batch.labels)
        s, t = union_topk_logprobs(
            h[keep],
            self.policy.lm_head,
            ht[keep],
            self.teacher.lm_head,
            k=cfg.topk,
            union=cfg.union_topk,
            with_tail=cfg.with_tail,
            chunk=cfg.logit_chunk,
        )
        return s, t, keep

    def loss(self, batch, denom: float) -> tuple[torch.Tensor, dict]:
        """Sum of per-token divergences, divided by the step's token count."""

        s, t, keep = self.supports(batch)
        per_token = divergence(s, t, self.cfg.jsd_beta)
        loss = per_token.sum() / denom
        return loss, {
            "tokens": int(keep.sum()),
            "kl": float(per_token.detach().sum()),
            "on_policy": float(sum(bool(m.get("on_policy")) for m in batch.meta)),
        }
