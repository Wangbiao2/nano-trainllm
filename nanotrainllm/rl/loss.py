"""The clipped policy-gradient objective, and the three ways to average it.

    ratio_t = exp(log p(y_t) - log p_old(y_t))
    L_t     = -min( ratio_t * A_t , clip(ratio_t, 1-eps_lo, 1+eps_hi) * A_t )

`min` is a switch, not a smoothing device: when the clipped branch wins its gradient
is exactly zero and that token's update is off.  That flat spot *is* the trust region
-- `clip_frac` counts how often it fires, and above ~0.2 the step is too large for `eps`.

With `num_iterations == 1` the sampling policy *is* the current policy, so `old_logp`
is `logp` detached, `ratio` is identically 1, the clip never fires and the loss
degenerates to `-A_t * log p(y_t)`: REINFORCE with a group baseline.  That is why the
second forward pass is skipped and `clip_frac` is always 0.

Aggregation changes the effective learning rate, which is why it is a flag and not a
constant.  `token-mean` weights tokens equally, so long completions dominate (DAPO's
choice); `seq-mean-token-sum` weights sequences equally but keeps their sums;
`seq-mean-token-mean` averages inside the sequence too, so length cancels (GRPO's own
choice, and the one DAPO argues against).  Each takes an explicit `denom` for the
reason `sft.py` does: the divisor must span the whole step.

`--policy-loss-type gspo` clips one length-normalised ratio per sequence,
`s = exp(mean_t(log p - log p_old))` -- a geometric mean, which barely strays from 1,
so `eps` drops two orders of magnitude and a completion is in or out as a whole.
`gspo-token` keeps that value but routes the gradient through each token's own `logp`.
`cispo` drops `min` altogether: the clipped ratio becomes a *detached* coefficient on
REINFORCE, so no token's gradient is switched off -- only down-weighted.
"""

from __future__ import annotations

import torch

SAFETY = 20.0

AGG_MODES = ("token-mean", "seq-mean-token-sum", "seq-mean-token-mean")

LOSS_TYPES = ("default", "gspo", "gspo-token", "cispo")


def denom_units(mask: torch.Tensor, mode: str) -> float:
    """How many things this aggregation mode divides by, for one micro-batch.

    `token-mean` divides by tokens, both `seq-mean-*` modes by sequences.  A
    sequence with no unmasked token adds nothing to the numerator, so it must not
    add to the denominator either.
    """
    assert mode in AGG_MODES, mode
    if mode == "token-mean":
        return float(mask.sum())
    return float((mask.sum(-1) > 0).sum())


def agg_loss(mat: torch.Tensor, mask: torch.Tensor, mode: str, denom: float) -> torch.Tensor:
    """Masked aggregation of a `[B, T]` per-token loss into a scalar.

    The divisor is passed in, not taken from this micro-batch's own mask -- the same
    global-denominator rule `sft.py` follows.
    """
    assert mode in AGG_MODES, mode
    assert mat.shape == mask.shape, (mat.shape, mask.shape)
    m = mask.to(mat.dtype)
    if mode == "token-mean":
        return (mat * m).sum() / max(denom, 1.0)
    per_seq = (mat * m).sum(-1)
    if mode == "seq-mean-token-mean":
        per_seq = per_seq / m.sum(-1).clamp(min=1.0)
    return per_seq.sum() / max(denom, 1.0)


def kl_penalty(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """Per-token estimate of `KL(policy || ref)` from one sample, Schulman's k3.

    With `d = log p_ref - log p_policy`, `exp(d) - d - 1` is unbiased, always `>= 0`,
    and lower variance than the obvious `-d` (also unbiased but signed, so a single
    sample can claim a negative KL).  `d**2 / 2` is the other one seen in the wild:
    non-negative but biased, being this expansion's second-order term.
    """
    d = (ref_logp - logp).clamp(-SAFETY, SAFETY)
    return d.exp() - d - 1.0


def policy_loss(
    logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    denom: float,
    old_logp: torch.Tensor | None = None,
    ref_logp: torch.Tensor | None = None,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    delta: float = -1.0,
    kl_coef: float = 0.0,
    loss_agg_mode: str = "token-mean",
    policy_loss_type: str = "default",
) -> tuple[torch.Tensor, dict]:
    """The scalar to backward, plus already-summed metrics.

    All shapes are `[B, T]` over the completion frame; `mask` is 1 on generated
    tokens, 0 on prompt and padding.  `advantages` is per-token (see
    `advantage.expand_to_per_token`) so OPD-RL needs no separate path.

    `delta > 0` additionally caps the *unclipped* branch (dual-clip PPO), bounding one
    enormously off-policy token when `A_t < 0` -- the case the standard clip does not
    cover, because `min` picks the unclipped branch there.

    Under `gspo` / `gspo-token` the ratio is one number per row, so both metrics change
    meaning without changing units: `ratio` sums `s_i` over the row's tokens and
    `clip_frac` counts the tokens *of* a clipped sequence rather than the clipped tokens
    of a sequence.  Both stay additive, which is what `loop.py` needs.

    Under `cispo` the objective is `-A_t * log p(y_t) * clip(ratio_t).detach()`, and
    three readings change.  `clip_frac` becomes "the coefficient was clamped", and may
    sit near 1.0 while the loss still falls -- the opposite of the default branch.
    `loss` is no longer comparable across branches: the default value is bounded by
    `|A|`, this one grows with `-log p`.  And `delta` has no unclipped branch to cap.
    """
    assert logp.shape == advantages.shape == mask.shape
    assert policy_loss_type in LOSS_TYPES, policy_loss_type
    m = mask.to(logp.dtype)
    on_policy = old_logp is None
    if on_policy:
        old_logp = logp.detach()

    log_ratio = (logp - old_logp) * m
    if policy_loss_type in ("gspo", "gspo-token"):
        # GSPO: one length-normalised ratio per sequence, `s = (p/p_old)^(1/|y|)`.
        # The mean runs along dim=-1, i.e. *inside* a row, which is sound only
        # because an on-policy stage forbids `--padding-free` (`stages/grpo.py`):
        # a row is exactly one completion, so no micro-batch boundary cuts through
        # the mean.  The opposite of `denom`, which has to be global.
        seq = log_ratio.sum(-1) / m.sum(-1).clamp(min=1.0)
        if policy_loss_type == "gspo-token":
            # Same value as `gspo` (the last two terms cancel numerically), but
            # d/d logp_t lands on that token alone: the trust region is decided
            # per sequence while credit is still assigned per token.
            log_ratio = (seq.detach()[:, None] + logp - logp.detach()) * m
        else:
            log_ratio = seq[:, None] * m
    log_ratio = log_ratio.clamp(-SAFETY, SAFETY)
    ratio = log_ratio.exp()

    clipped = ratio.clamp(1.0 - epsilon_low, 1.0 + epsilon_high)
    if policy_loss_type == "cispo":
        # The clipped ratio is a *detached coefficient* on REINFORCE, not a factor
        # inside a `min`.  Nothing is switched off, so the rare low-probability
        # token that decides where a chain of thought turns gets down-weighted
        # instead of discarded -- the whole point of CISPO.
        assert delta <= 0, "--delta has no unclipped branch to bound under cispo"
        # The one branch that reads `logp` raw rather than as a difference, so it
        # relies on `stages/grpo.py::sequence_logprobs` zero-filling outside `scored`:
        # `agg_loss` masks the result, but 0 * nan is still nan.
        per_token = -advantages * logp * clipped.detach()
        # No branch switch to count; `clamp` returns its input untouched inside the
        # interval, so this equality is exact rather than a tolerance test.
        bound = (clipped != ratio) & mask.bool()
    else:
        unclipped = ratio if delta <= 0 else ratio.clamp(max=delta)
        obj_unclipped = unclipped * advantages
        obj_clipped = clipped * advantages
        per_token = -torch.min(obj_unclipped, obj_clipped)
        # The clip *binds* exactly when the clipped branch is the smaller objective;
        # that is the definition, and it is also where the gradient becomes zero.
        bound = (obj_clipped < obj_unclipped) & mask.bool()

    metrics = {
        "advantage": float((advantages * m).sum()),
        "ratio": float((ratio.detach() * m).sum()),
        "clip_frac": float(bound.sum()),
    }

    if kl_coef != 0.0:
        assert ref_logp is not None, "--kl-coef needs a reference model"
        kl = kl_penalty(logp, ref_logp)
        per_token = per_token + kl_coef * kl
        metrics["kl"] = float((kl.detach() * m).sum())

    loss = agg_loss(per_token, mask, loss_agg_mode, denom)
    metrics["tokens"] = float(mask.sum())
    return loss, metrics
