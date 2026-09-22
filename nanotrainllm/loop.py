"""The training loop, written out longhand.

No Trainer class from a library, no callbacks: `for step: for micro_batch:
backward` plus a cosine schedule, because the interesting post-training bugs live
here rather than in the model.  Three rules it exists to enforce:

1. **The denominator is computed before any backward.**  `stage.denom(batches)`
   sees every micro-batch of the step and reduces across ranks, so no
   `loss / grad_accum` appears below and `micro=1 x accum=4` gives exactly the
   gradient of `micro=4 x accum=1`.
2. **The step count is fixed up front**, because the LR schedule needs to know
   where the end is; a ragged final step is dropped rather than run short.
3. **`reset_rope_cache()` after every rollout**, so a stale `model.rope_deltas`
   written by `generate()` can never leak into a training forward.
"""

from __future__ import annotations

import math
import os

import torch
from tqdm import tqdm

from .stages import build


def build_optimizer(cfg, policy) -> torch.optim.Optimizer:
    """AdamW over exactly the parameters `model.py` left trainable.

    No parameter groups: the default is `weight_decay=0.0`, and a grouping that only
    ever divides zero by two is decoration.
    """
    params = policy.trainable()
    assert params, "nothing to train; check --tuner / --freeze-* flags"
    return torch.optim.AdamW(
        params, lr=cfg.lr, betas=tuple(cfg.betas), weight_decay=cfg.weight_decay
    )


def lr_at(cfg, step: int, total: int) -> float:
    """Linear warmup then cosine decay to `lr * lr_min_ratio`.

    `step` is 0-based and warmup runs at `(step + 1) / warm`, so the first step is not
    at lr=0 -- which would waste it.
    """
    warm = min(total, max(1, round(total * cfg.warmup_ratio)))
    if step < warm:
        return cfg.lr * (step + 1) / warm
    t = (step - warm) / max(1, total - warm)
    lo = cfg.lr * cfg.lr_min_ratio
    return lo + (cfg.lr - lo) * 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))


class Trainer:
    def __init__(self, cfg, policy, dataset, **collaborators):
        self.cfg = cfg
        self.policy = policy
        self.dataset = dataset
        self.stage = build(cfg, policy=policy, **collaborators)
        self.opt = build_optimizer(cfg, policy)
        self.history: list[dict] = []
        if cfg.is_on_policy:
            self._check_on_policy_batching()

    # -- planning ----------------------------------------------------------
    def _check_on_policy_batching(self) -> None:
        """Cross-rank lockstep checks for the rollout/global/micro schedule.

        Not in `Config.__post_init__` because it needs the real `world_size`, which
        `cli.init_distributed` fills in after construction.  Every rank must run the
        same number of rollouts and updates, or the `all_reduce` inside `denom`
        deadlocks on a ragged tail.
        """
        cfg = self.cfg
        w, micro = cfg.world_size, cfg.micro_batch_size
        gp, rp = cfg.global_prompts, cfg.rollout_prompts
        assert gp % w == 0, (
            f"global_batch_size={gp} (prompts) must be divisible by world_size={w}"
        )
        # A forward micro-batch is `micro_batch_size` *completions*, so it is the
        # per-rank completion count of an update that must divide by it, not the
        # prompt count.  `num_generations` completions come from each prompt.
        g_comps = (gp // w) * cfg.num_generations
        assert g_comps % micro == 0, (
            f"per-rank completions per update = (global_batch_size/world)*num_generations "
            f"= {g_comps} must be divisible by micro_batch_size={micro}"
        )
        assert rp % gp == 0, (
            f"rollout_batch_size={rp} must be a multiple of global_batch_size={gp}"
        )
        assert rp % w == 0, f"rollout_batch_size={rp} must be divisible by world_size={w}"
        assert self._rollouts_per_epoch() >= 1, (
            f"the dataset yields {len(self.dataset.batches(0))} micro-batches per rank, "
            f"fewer than the {self._units_per_rollout()} one rollout needs "
            f"(rollout_batch_size={rp}, world={w}); "
            "lower --rollout-batch-size or add data"
        )

    def _units_per_rollout(self) -> int:
        """Dataset prompts (one per unit, on-policy) one rollout draws per rank."""
        cfg = self.cfg
        return cfg.rollout_prompts // cfg.world_size

    def _rollouts_per_epoch(self) -> int:
        return len(self.dataset.batches(0)) // self._units_per_rollout()

    def steps_per_epoch(self) -> int:
        cfg = self.cfg
        if cfg.is_on_policy:
            # one optimiser step per (rollout, ppo-epoch, mini-batch)
            return self._rollouts_per_epoch() * cfg.num_iterations * cfg.updates_per_rollout
        return len(self.dataset.batches(0)) // cfg.grad_accum

    def total_steps(self) -> int:
        n = self.steps_per_epoch() * self.cfg.epochs
        assert n > 0, (
            f"{len(self.dataset)} groups give fewer than one optimiser step at "
            f"micro_batch_size={self.cfg.micro_batch_size} x grad_accum={self.cfg.grad_accum}"
        )
        return n if self.cfg.max_steps < 0 else min(n, self.cfg.max_steps)

    # -- one optimiser step ------------------------------------------------
    def _update(self, batches: list, index: int, total: int) -> dict:
        """The optimiser step over an already-built list of micro-batches.

        `denom` spans exactly these micro-batches and is reduced across ranks, which
        is what makes `micro x accum` and `accum x micro` agree.
        """
        cfg = self.cfg
        denom = self.stage.denom(batches)

        lr = lr_at(cfg, index, total)
        for g in self.opt.param_groups:
            g["lr"] = lr

        metrics = {"loss": 0.0}
        for batch in batches:
            loss, extra = self.stage.loss(batch, denom)
            loss.backward()
            metrics["loss"] += float(loss.detach())
            for k, v in extra.items():
                metrics[k] = metrics.get(k, 0.0) + v

        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.trainable(), cfg.max_grad_norm)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        metrics.update(lr=lr, grad_norm=float(grad_norm), denom=denom)
        return metrics

    def step(self, units: list[list], index: int, total: int) -> dict:
        """Off-policy step: build this step's micro-batches, then update once.

        pt/sft/dpo, plus opd (which mixes on- and off-policy inside its own
        `micro_batches`).  On-policy stages take the two-tier path in `run`.
        """
        batches = self.stage.micro_batches(units)
        if self.cfg.needs_rollout:
            self.policy.reset_rope_cache()
        return self._update(batches, index, total)

    # -- the loop ----------------------------------------------------------
    def run(self) -> list[dict]:
        cfg = self.cfg
        total = self.total_steps()
        bar = tqdm(total=total, disable=cfg.rank != 0, dynamic_ncols=True, desc=cfg.stage)
        loop = self._loop_on_policy if cfg.is_on_policy else self._loop_standard
        index = loop(bar, total)
        bar.close()
        self.save(index)
        return self.history

    def _record(self, bar, index: int, epoch: int, m: dict) -> None:
        """History append + tqdm postfix + periodic save.  `index` is 1-based."""
        cfg = self.cfg
        self.history.append({"step": index, "epoch": epoch, **m})
        if index % cfg.log_every == 0:
            post = dict(
                loss=f"{m['loss']:.4f}",
                lr=f"{m['lr']:.2e}",
                gnorm=f"{m['grad_norm']:.2f}",
            )
            # `reward` is summed over the step's completions; show it per completion,
            # plus each rule's own mean so a run that learns only `format` is visible.
            comps = m.get("completions", 0.0)
            if comps > 0:
                for k, v in m.items():
                    if k == "reward" or k.startswith("reward/"):
                        post[k] = f"{v / comps:.3f}"
            bar.set_postfix(**post)
        bar.update(1)
        if cfg.save_every > 0 and index % cfg.save_every == 0:
            self.save(index)

    def _loop_standard(self, bar, total: int) -> int:
        """pt/sft/dpo/opd: one optimiser step per `grad_accum` micro-batches."""
        cfg = self.cfg
        per_epoch = self.steps_per_epoch()
        index = 0
        for epoch in range(cfg.epochs):
            units = self.dataset.batches(epoch)
            for s in range(per_epoch):
                if index >= total:
                    break
                group = units[s * cfg.grad_accum : (s + 1) * cfg.grad_accum]
                m = self.step(group, index, total)
                index += 1
                self._record(bar, index, epoch, m)
        return index

    def _loop_on_policy(self, bar, total: int) -> int:
        """grpo/opd-rl: build a rollout buffer, then replay it as mini-batches.

        Each rollout draws `_units_per_rollout()` prompts and `stage.micro_batches`
        turns them into a flat list of collated forward batches -- generating,
        scoring, whitening, and (when off-policy) caching `old_logp` *before* any
        optimiser step.  That buffer is sliced into `updates_per_rollout`
        mini-batches and replayed `num_iterations` times.

        `total_steps` is world-deterministic, so every rank builds the same number of
        buffers and runs the same number of updates: `denom`'s `all_reduce` never
        waits on a ragged tail.
        """
        cfg = self.cfg
        rollouts = self._rollouts_per_epoch()
        units_per_rollout = self._units_per_rollout()
        # Forward micro-batches (of `micro_batch_size` completions) per rank per update.
        mb_per_update = (
            cfg.global_prompts // cfg.world_size * cfg.num_generations
        ) // cfg.micro_batch_size
        index = 0
        for epoch in range(cfg.epochs):
            units = self.dataset.batches(epoch)
            for r in range(rollouts):
                if index >= total:
                    break
                chunk = units[r * units_per_rollout : (r + 1) * units_per_rollout]
                buffer = self.stage.micro_batches(chunk)
                self.policy.reset_rope_cache()
                # `micro_batches` stashes the rollout's statistics in `stage._logged`
                # for the loss to fold in once.  Every update trains on the replayed
                # buffer, so each re-arms them -- otherwise only the first reports.
                step_logged = dict(self.stage._logged)
                done = False
                for _ppo in range(cfg.num_iterations):
                    for u in range(cfg.updates_per_rollout):
                        if index >= total:
                            done = True
                            break
                        self.stage._logged = dict(step_logged)
                        mini = buffer[u * mb_per_update : (u + 1) * mb_per_update]
                        m = self._update(mini, index, total)
                        index += 1
                        self._record(bar, index, epoch, m)
                    if done:
                        break
        return index

    # -- checkpointing -----------------------------------------------------
    def save(self, step: int) -> str | None:
        """Rank 0 writes `state_dict()` -- LoRA deltas only, unless `--tuner full`.

        The all-gather inside `state_dict()` is a collective, so it runs *before* the
        rank-0 guard: skipping it elsewhere would hang rank 0 on a gather nobody
        joined.
        """
        sd = self.policy.state_dict()
        if self.cfg.rank != 0:
            return None
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        path = os.path.join(self.cfg.output_dir, f"step-{step}.pt")
        torch.save(sd, path)
        return path
