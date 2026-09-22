"""From one scalar reward per completion to one advantage per token.

GRPO has no value network: PPO's learned baseline becomes the mean reward of the
*group*, the `G` completions of one prompt.  Hence `num_generations >= 2` in
`config.py` -- with G=1 the mean is the sample and every advantage is zero.

    A_i = (r_i - mean(r_group)) / (std(r_group) + eps)

Dividing by the group std whitens per prompt, so the update is invariant to how
hard the prompt is (the original GRPO paper).  It also amplifies groups where all
G scored nearly the same -- (1, 1, 1, 0) gets the same |A| as (1, 0, 0, 0), pure
variance injection on a noisy reward.  Dr.GRPO drops it: `--scale-rewards false`.

RLOO swaps the group mean for the leave-one-out mean `b_i = (S - r_i)/(G-1)`, so a
completion is never in its own baseline.  Expand it: `(G/(G-1))*(r_i - mean_G)` --
with `G` fixed and outcome-only rewards that constant is all RLOO adds, i.e. a
learning-rate change (x2 at G=2, x1.03 at G=32).

The per-token expansion lives here rather than in the loss, because OPD-RL adds a real
per-token term on top of the broadcast constant (`stages/opd_rl.py`); with that term off,
`expand_to_per_token` is a plain broadcast and the result is exactly GRPO.
"""

from __future__ import annotations

import torch

# exp() of anything larger overflows fp32 well before it means anything; the
# clamp only fires when a policy has already diverged.
SAFETY = 20.0

# Guards the whitening when a group's rewards are all equal, so `std == 0`.  Small
# enough not to shrink a real spread, large enough that 0/eps stays finite.
STD_EPS = 1e-4

# Which baseline the group's reward is measured against.  Both are value-free.
ESTIMATORS = ("grpo", "rloo")


def group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    scale_rewards: bool = True,
    estimator: str = "grpo",
) -> torch.Tensor:
    """`rewards [B*G]`, group-major -> advantages `[B*G]`.

    Group-major means completion `j` of prompt `i` sits at `i * G + j`, the order
    `rollout.generate` returns, so a `[B, G]` view *is* the grouping.  `std` is the
    unbiased (n-1) estimator, zero when a group scored uniformly -- hence `STD_EPS`.

    Whitening does *not* cancel RLOO's `G/(G-1)`: the divisor is the std of the
    rewards, not of the advantages, so the constant survives either way.
    """
    assert rewards.ndim == 1, rewards.shape
    assert rewards.shape[0] % group_size == 0, (rewards.shape, group_size)
    assert estimator in ESTIMATORS, estimator
    r = rewards.view(-1, group_size).float()
    if estimator == "rloo":
        # `b_i = (S - r_i) / (G - 1)`: completion `i` is scored against the *other*
        # G-1 only.  An unbiased baseline in general; here, a constant rescale.
        assert group_size > 1, "rloo needs num_generations >= 2"
        adv = r - (r.sum(-1, keepdim=True) - r) / (group_size - 1)
    else:
        adv = r - r.mean(-1, keepdim=True)
    if scale_rewards:
        std = r.std(-1, keepdim=True) if group_size > 1 else torch.zeros_like(adv)
        adv = adv / (std + STD_EPS)
    return adv.reshape(-1)


def teacher_logratio(
    teacher_logp: torch.Tensor,
    policy_logp: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """`log p_T(y_t) - log p_S(y_t)` on the sampled tokens, zero outside the mask.

    A difference of two scalars per position -- both sides score the *same* token --
    not a KL over the vocabulary.  Masking comes before the subtraction: the padding
    positions hold whatever the collator left there.
    """
    return (teacher_logp - policy_logp).masked_fill(~mask.bool(), 0.0)


def teacher_kl(
    teacher_logp: torch.Tensor,
    policy_logp: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """k3 estimator of `KL(S || T)` per token -- monitoring only, never a loss here.

    `exp(d) - d - 1 >= 0`, so as a reward it could only say "avoid disagreeing", never
    which direction to move.  That is why the advantage uses the signed k1 instead.
    """
    d = (teacher_logp - policy_logp).masked_fill(~mask.bool(), 0.0).clamp(-SAFETY, SAFETY)
    return (d.exp() - d - 1.0) * mask


def expand_to_per_token(
    advantages: torch.Tensor,
    mask: torch.Tensor,
    teacher_logp: torch.Tensor | None = None,
    policy_logp: torch.Tensor | None = None,
    teacher_kl_coef: float = 0.0,
) -> torch.Tensor:
    """`advantages [B]` -> `[B, T]`, plus the OPD-RL teacher term when asked for.

    With `teacher_kl_coef == 0` (or no teacher) this is a plain broadcast, constant
    along `T` -- exactly GRPO.
    """
    assert advantages.ndim == 1 and mask.ndim == 2
    assert advantages.shape[0] == mask.shape[0], (advantages.shape, mask.shape)
    per_token = advantages[:, None].expand_as(mask).clone()
    if teacher_logp is None or teacher_kl_coef == 0.0:
        return per_token
    assert policy_logp is not None, "the teacher term needs the student's logp too"
    return per_token + teacher_kl_coef * teacher_logratio(teacher_logp, policy_logp, mask)
