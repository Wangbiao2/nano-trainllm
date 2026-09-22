"""GRPO: the five steps, in the order they happen.

Everything hard about GRPO is already in `rl/`; this file is the wiring that
answers the three `Stage` questions so `loop.py` stays stage-agnostic.  One
optimiser step is:

  1. **generate** -- `rollout.generate(prompts, G)` returns `len(prompts) * G`
     completions, group-major (`prompt i, sample j` at `i * G + j`).
  2. **score** -- `reward.compute_rewards` turns the decoded text into `[B*G]`.
  3. **whiten** -- `advantage.group_advantages` centres each group of `G` by its
     own mean, so no value network is needed, then broadcasts to tokens.
  4. **surrogate** -- `loss.policy_loss` with `ratio = 1` (see `num_iterations`).
  5. **KL** -- optional `k3` penalty against a frozen reference.

Generation happens in `micro_batches`, not in `loop.py`: an on-policy stage's
"dataset" is produced by the model itself, and it has to exist before the
denominator can count the generated tokens.  That is why `Stage` has three methods.

The advantage is computed once over the whole step and carried in `meta`: whitening
per micro-batch is a different (and wrong) algorithm once a step spans several
prompts, because the baseline must be the group mean.
"""

from __future__ import annotations

import torch

from ..collate import collate
from ..logprobs import shift_for_causal_lm, token_logprobs
from ..rl.advantage import expand_to_per_token, group_advantages
from ..rl.loss import denom_units, policy_loss
from ..rl.reward import REWARDS, compute_rewards
from ..rl.rollout import completion_mask, to_encoded
from . import Stage, all_sum


def sequence_logprobs(policy, batch, scored: torch.Tensor, cfg) -> torch.Tensor:
    """`log p(y_t)` on the generated tokens -> `[B, S-1]`, zero elsewhere.

    `scored` selects the positions to evaluate: the LM head is the expensive part
    (~250k columns), and `labels` is `-100` outside them, which is not a valid gather
    index.

    Scoring happens at `cfg.temperature`, not at 1.0 -- the distribution being
    optimised *is* the tempered one the sampler drew from, and any other temperature
    silently biases the importance ratio and the KL against the reference.
    """
    hidden = policy.hidden(batch)
    h, y = shift_for_causal_lm(hidden, batch.labels)
    keep = scored.reshape(-1).bool()
    lp = token_logprobs(h[keep], policy.lm_head, y[keep], cfg.temperature, cfg.logit_chunk)
    out = torch.zeros(scored.numel(), dtype=lp.dtype, device=lp.device)
    out[keep] = lp
    return out.view_as(scored)


class GRPO(Stage):
    def __init__(self, cfg, policy, ref=None, teacher=None, rollout=None):
        super().__init__(cfg, policy, ref, teacher, rollout)
        assert rollout is not None, "an on-policy stage needs a rollout"
        # The `[B, G]` view in `group_advantages` and the per-sequence aggregation
        # in `agg_loss` both want one row per completion; a padding-free batch
        # holds several completions per row and neither would mean anything.
        assert not cfg.padding_free, "grpo needs one row per completion (--padding-free off)"
        # Off-policy PPO: the buffer is reused across `num_iterations` epochs and/or
        # split into `updates_per_rollout` steps, so a token's sampling policy is no
        # longer the current one and the clipped ratio in `rl/loss.py` goes live on
        # the `old_logp` cached below.  One update over one pass => ratio == 1.
        self.off_policy = (cfg.num_iterations * cfg.updates_per_rollout) > 1
        if cfg.stage == "opd-rl":
            assert not self.off_policy, (
                "opd-rl's teacher term uses the *current* policy's log-probs and has no "
                "off-policy correction; keep --num-iterations 1 and "
                "--global-batch-size == --rollout-batch-size"
            )
        if cfg.kl_coef != 0.0:
            assert ref is not None, "--kl-coef needs a reference model"
        self._logged: dict = {}
        self._rollouts_seen = 0  # gates the rank-0 sample dump (--log-samples-every)

    # -- 1..3: generate, score, whiten ------------------------------------
    def micro_batches(self, units: list[list]) -> list:
        cfg, n = self.cfg, self.cfg.num_generations
        prompts = [e for u in units for e in u]
        completions = self.rollout.generate(prompts, n)
        assert len(completions) == len(prompts) * n, "rollout must be group-major"

        refs = [_reference(c.prompt.meta) for c in completions]
        lens = [len(c) for c in completions]
        rewards, logged = compute_rewards([c.text for c in completions], refs, lens, cfg)
        adv = group_advantages(rewards, n, cfg.scale_rewards, cfg.advantage_estimator)

        self._rollouts_seen += 1
        every = cfg.log_samples_every
        if cfg.rank == 0 and every and self._rollouts_seen % every == 0:
            _log_samples(completions, refs, rewards, lens, cfg, self._rollouts_seen)

        rows = []
        for c, a in zip(completions, adv.tolist()):
            e = to_encoded(c)
            e.meta["advantage"] = a
            rows.append(e)

        # Rollout statistics belong to the step, not to a micro-batch, so they are
        # stashed here and folded into the first `loss` call's metrics.
        self._logged = {
            **logged,
            "completions": float(len(completions)),
            "completion_length": float(sum(lens)),
            "truncated": float(sum(c.truncated for c in completions)),
        }
        # Forward micro-batches are `micro_batch_size` completions sliced in order,
        # decoupled from the group size `n`, so one may span prompts and groups.  That
        # is sound because the advantage is per-completion (in `meta`), every row has
        # its own attention mask, and `denom` is global.  The tail must divide evenly
        # or a group would straddle the loop's group-aligned update boundaries.
        mb = cfg.micro_batch_size
        assert len(rows) % mb == 0, (
            f"{len(rows)} completions (prompts*G) not divisible by "
            f"micro_batch_size={mb}; a forward micro-batch would split a group"
        )
        batches = [collate(rows[i * mb : (i + 1) * mb], cfg) for i in range(len(rows) // mb)]
        # The sampling policy is *this* policy, right now, before any optimiser step
        # touches it -- so log p_old is cached here and reused for every replay.
        # `advantage` depends only on rewards, so nothing else goes stale.
        if self.off_policy:
            for b in batches:
                scored = completion_mask(b, self.policy.device)
                with torch.no_grad():
                    b.old_logp = sequence_logprobs(self.policy, b, scored, cfg)
        return batches

    # -- the divisor ------------------------------------------------------
    def denom(self, batches: list) -> float:
        """Tokens or sequences in the whole step, per `--loss-agg-mode`.

        Counted here rather than inside `agg_loss` so all three aggregation modes stay
        additive across micro-batches and ranks.
        """
        mode = self.cfg.loss_agg_mode
        n = sum(denom_units(self._loss_mask(b, completion_mask(b)), mode) for b in batches)
        return max(all_sum(n, device=self.policy.device), 1.0)

    def _loss_mask(self, batch, scored: torch.Tensor) -> torch.Tensor:
        """`scored`, minus any completion `--overlong-filter` disqualifies.

        A completion cut off at `max_completion_length` is not a fair sample of the
        policy: training on its (usually low) reward teaches that long answers are
        bad.  Dropping it from the loss rather than from the group keeps the group mean
        honest -- the truncation is still evidence about the prompt, it just gets no
        gradient.
        """
        if not self.cfg.overlong_filter:
            return scored
        keep = torch.tensor(
            [not m.get("truncated", False) for m in batch.meta],
            device=scored.device,
            dtype=scored.dtype,
        )
        return scored * keep[:, None]

    # -- 4..5: surrogate and KL -------------------------------------------
    def advantages(self, batch, mask, adv, logp) -> torch.Tensor:
        """Hook: `[B]` -> `[B, T]`.  `opd_rl.py` overrides it to add its own term."""
        return expand_to_per_token(adv, mask)

    def loss(self, batch, denom: float) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        scored = completion_mask(batch, self.policy.device)
        logp = sequence_logprobs(self.policy, batch, scored, cfg)
        mask = self._loss_mask(batch, scored)
        adv = torch.tensor(
            [m["advantage"] for m in batch.meta], device=logp.device, dtype=logp.dtype
        )

        ref_logp = None
        if cfg.kl_coef != 0.0:
            with torch.no_grad():
                ref_logp = sequence_logprobs(self.ref, batch, scored, cfg)

        loss, metrics = policy_loss(
            logp,
            self.advantages(batch, mask, adv, logp),
            mask,
            denom,
            old_logp=batch.old_logp,
            ref_logp=ref_logp,
            epsilon_low=cfg.epsilon_low,
            epsilon_high=cfg.epsilon_high,
            delta=cfg.delta,
            kl_coef=cfg.kl_coef,
            loss_agg_mode=cfg.loss_agg_mode,
            policy_loss_type=cfg.policy_loss_type,
        )
        metrics.update(self._logged)
        self._logged = {}  # one micro-batch pays for them; the loop sums the rest
        return loss, metrics


_W = 78  # width of the header / footer rule, in characters


def _log_samples(completions, refs, rewards, lens, cfg, rollout: int) -> None:
    """Rank-0 dump of one complete sample every `--log-samples-every` rollouts.

    One sample rather than the whole group, and printed in full rather than clipped,
    so a run can be read straight from the log.  The per-rule breakdown beside the
    total is what separates "the model is wrong" from "the reward cannot parse the
    answer".
    """
    names, weights = cfg.reward_func_names, cfg.reward_weight_values
    c = completions[0]  # first completion of the first group in the rollout
    out_tok, in_tok = lens[0], len(c.prompt.input_ids)
    parts = "  ".join(
        f"{name} {w * REWARDS[name](c.text, refs[0], cfg, out_tok):+.3f}"
        for name, w in zip(names, weights)
    )
    trunc = "  (truncated)" if c.truncated else ""
    lines = [
        "",
        f"┌─ rollout {rollout}  group 0  sample 0 " + "─" * _W,
        f"[Q]      {str(c.prompt.meta.get('question', '')).strip()}",
        f"[GT]     {str(refs[0]).strip()}",
        f"[answer] {c.text.strip()}",
        f"[reward] {parts}   =>  total {rewards[0].item():+.3f}",
        f"[length] input {in_tok} tok  /  output {out_tok} tok{trunc}",
        "└" + "─" * _W,
    ]
    print("\n".join(lines))


def _reference(meta: dict) -> str | None:
    """The ground-truth answer a rule reward compares against.

    `template.encode_prompt` copies whichever of these keys the row had, so this is
    where "what column holds the answer" is decided.  None is fine: `reward.accuracy`
    scores an unanswerable prompt 0.0 for every completion, which whitens to an
    all-zero advantage rather than to noise.
    """
    for key in ("solution", "answer", "label"):
        if key in meta:
            return meta[key]
    return None
