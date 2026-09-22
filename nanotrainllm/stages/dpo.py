"""DPO: no reward model, no sampling -- just the gap between two log-probabilities.

`max_pi E[r] - beta * KL(pi || pi_ref)` has the closed-form optimum `pi* =
pi_ref exp(r/beta) / Z(x)`, so *any* policy implicitly defines a reward `r = beta
* log(pi / pi_ref) + beta * log Z(x)`.  Feed that into the Bradley-Terry
likelihood of `y_w > y_l` and `Z(x)` cancels -- it depends only on the prompt:

    L = -log sigmoid( beta * [ (log pi(y_w) - log pi_ref(y_w))
                             - (log pi(y_l) - log pi_ref(y_l)) ] )

Four sequence log-probabilities, no sampling, no value network, no reward model.
The rest is bookkeeping to line those four numbers up, and four parts of it are
easy to get wrong:

* **A pair must live inside one forward pass.**  `encode_pair` emits the two rows
  adjacently and `dataset.py` shuffles *groups*, so `[0::2]` / `[1::2]` recover
  them.  Split across an accumulation boundary the difference would be taken
  between two different forward passes -- wrong, and nothing would report it.
* **The log-probability is a *sum*, not a mean.**  Bradley-Terry is over whole
  sequences; dividing by the length changes the objective (which is what `ipo`
  implicitly wants).  `sequence_logps` sums, and returns the counts separately.
* **The reference can be the policy itself.**  Under `--tuner lora` the base
  weights are untouched, so `Policy.no_adapter()` *is* `pi_ref` -- exact and
  free.  A full-parameter run has overwritten `W` and needs a real second model.
* **The denominator counts pairs**, not tokens: the loss is one number per pair.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..logprobs import shift_for_causal_lm, token_logprobs
from ..template import IGNORE
from . import Stage, all_sum


def sequence_ids(batch) -> torch.Tensor:
    """`[B, S-1]` -> which sample of the batch each shifted position belongs to.

    Just the row index in the padded layout.  Packed and padding-free rows hold
    several samples end to end, and the shift means column `i` belongs to whatever
    sample token `i + 1` came from -- which is what `[:, 1:]` gives.
    """
    b, s = batch.labels.shape
    device = batch.labels.device
    if not batch.padding_free:
        return torch.arange(b, device=device)[:, None].expand(b, s)[:, 1:]
    lens = torch.tensor(batch.seq_lens, device=device)
    ids = torch.repeat_interleave(torch.arange(len(batch.seq_lens), device=device), lens)
    assert ids.numel() == b * s, f"{int(lens.sum())} packed tokens in a [{b}, {s}] batch"
    return ids.view(b, s)[:, 1:]


def sequence_logps(policy, batch, cfg) -> tuple[torch.Tensor, torch.Tensor]:
    """-> `([N], [N])`: summed log p of each sample's supervised tokens, and how many.

    Scored at temperature 1.0, unlike `grpo.sequence_logprobs`: there is no sampler to
    be consistent with, and `beta` already plays the temperature's role in the implicit
    reward.
    """
    h, y = shift_for_causal_lm(policy.hidden(batch), batch.labels)
    keep = y != IGNORE
    lp = token_logprobs(h[keep], policy.lm_head, y[keep], 1.0, cfg.logit_chunk)
    seg = sequence_ids(batch).reshape(-1)[keep]
    n = len(batch.seq_lens)
    total = torch.zeros(n, dtype=lp.dtype, device=lp.device).index_add_(0, seg, lp)
    count = torch.zeros(n, dtype=lp.dtype, device=lp.device).index_add_(
        0, seg, torch.ones_like(lp)
    )
    assert float(count.min()) > 0.0, "a dpo row with nothing supervised"
    return total, count


def preference_loss(
    chosen: torch.Tensor,
    rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`[P]` log-probs -> `([P] loss, [P] chosen reward, [P] rejected reward)`.

    Everything is a function of one quantity, the implicit reward margin over `beta`:

        logits = (log pi(y_w) - log pi_ref(y_w)) - (log pi(y_l) - log pi_ref(y_l))

    The gradient of `-logsigmoid(beta * logits)` carries `sigmoid(-beta * logits)`,
    large exactly where the implicit reward model still ranks the pair wrongly and
    decaying to zero once it is comfortably right.  That self-annealing is why DPO
    needs no KL term: `beta` *is* the KL strength.  (Variants replace this one line:
    IPO squares `logits - 1/(2 beta)`, SLiC's hinge is `relu(1 - beta * logits)`, cDPO
    mixes in the flipped label.)

    The two returned rewards are the implicit `r(x, y)` up to the cancelling
    `beta * log Z(x)`; `chosen_reward > rejected_reward` is the quoted "accuracy".
    """
    chosen_reward = beta * (chosen - ref_chosen)
    rejected_reward = beta * (rejected - ref_rejected)
    logits = (chosen - ref_chosen) - (rejected - ref_rejected)
    return -F.logsigmoid(beta * logits), chosen_reward, rejected_reward


class DPO(Stage):
    def __init__(self, cfg, policy, ref=None, teacher=None, rollout=None):
        super().__init__(cfg, policy, ref, teacher, rollout)
        if not cfg.dpo_reference_free:
            assert ref is not None or cfg.tuner == "lora", (
                "dpo needs a reference model: pass one, or use --tuner lora and let "
                "Policy.no_adapter() supply it"
            )

    # -- the divisor -------------------------------------------------------
    def denom(self, batches: list) -> float:
        """Preference pairs in this optimiser step, summed over every rank."""
        n = sum(_pairs(b) for b in batches)
        return max(all_sum(n, device=self.policy.device), 1.0)

    # -- the loss ----------------------------------------------------------
    def reference_logps(self, batch) -> torch.Tensor:
        """`pi_ref`, from whichever of the three sources this run has.

        `--dpo-reference-free` drops the term (CPO's simple-preference objective).  It
        saves a forward and behaves worse: with no `pi_ref` to anchor it, nothing stops
        the policy from driving *both* log-probs down as long as the rejected one falls
        faster.
        """
        cfg = self.cfg
        if cfg.dpo_reference_free:
            return torch.zeros(len(batch.seq_lens), device=self.policy.device)
        if self.ref is not None:
            with torch.no_grad():
                return sequence_logps(self.ref, batch, cfg)[0]
        with torch.no_grad(), self.policy.no_adapter():
            return sequence_logps(self.policy, batch, cfg)[0]

    def loss(self, batch, denom: float) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        pairs = _pairs(batch)
        logps, counts = sequence_logps(self.policy, batch, cfg)
        ref = self.reference_logps(batch)

        losses, chosen_reward, rejected_reward = preference_loss(
            logps[0::2],
            logps[1::2],
            ref[0::2],
            ref[1::2],
            beta=cfg.dpo_beta,
        )
        loss = losses.sum() / denom

        nll = -(logps[0::2] / counts[0::2])
        if cfg.dpo_sft_weight != 0.0:
            # RPO: an NLL term on the chosen response.  DPO only constrains the
            # *difference* of two log-probs, so a policy can satisfy it while making
            # the good answer less likely too; this anchors it.  Length-normalised, or
            # one long chosen response would dominate a per-pair objective.
            loss = loss + cfg.dpo_sft_weight * nll.sum() / denom

        return loss, {
            "pairs": float(pairs),
            "tokens": int(counts.sum()),
            "chosen_logp": float(logps[0::2].detach().sum()),
            "rejected_logp": float(logps[1::2].detach().sum()),
            "chosen_reward": float(chosen_reward.detach().sum()),
            "rejected_reward": float(rejected_reward.detach().sum()),
            "margin": float((chosen_reward - rejected_reward).detach().sum()),
            "accuracy": float((chosen_reward > rejected_reward).sum()),
            "nll": float(nll.detach().sum()),
        }


def _pairs(batch) -> int:
    """How many preference pairs a batch holds, checking the layout while at it.

    The `[0::2]` / `[1::2]` recovery is only correct if the sides really do
    alternate, and the one thing that would break it -- a batch built from
    something other than `encode_pair` -- is silent otherwise: the loss would
    still be a finite number, computed on mismatched pairs.
    """
    sides = [m.get("side") for m in batch.meta]
    assert len(sides) == len(batch.seq_lens), "one meta per sample, in sample order"
    assert len(sides) % 2 == 0, f"dpo needs an even number of rows, got {len(sides)}"
    assert sides[0::2] == ["chosen"] * (len(sides) // 2), f"chosen/rejected not paired: {sides}"
    assert sides[1::2] == ["rejected"] * (len(sides) // 2), f"chosen/rejected not paired: {sides}"
    return len(sides) // 2
