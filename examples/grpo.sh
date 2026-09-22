#!/usr/bin/env bash
# GRPO: the model generates its own training data, and the group is the baseline.
#
# `--num-generations 8` is the G in "group": eight completions per prompt, whose
# reward mean is the value baseline -- which is why GRPO needs no critic and why
# G must be at least 2.  A group where every completion earns the same reward has
# zero advantage and contributes exactly nothing to the gradient; that is correct
# behaviour, not a bug, and it is what makes prompt difficulty matter.
#
# `--rollout vllm` is the fast path (weights synced via `Policy.merged_adapter()`
# between steps, engine put to sleep in between).  It is left off here because
# the HF `generate` backend needs no second engine and is the one you can step
# through in a debugger.
#
# `--scale-rewards` (default) divides each group's advantage by the group's reward
# std.  `--no-scale-rewards` is Dr.GRPO: the division whitens per prompt, but it
# also amplifies groups where everything scored nearly the same -- (1,1,1,0) gets
# the same |A| as (1,0,0,0), which is variance injection on a noisy reward.
#
# ---- variants, each one flag ------------------------------------------------
#
# `--advantage-estimator rloo` swaps the group mean for the leave-one-out mean, so a
# completion is not part of its own baseline.  It gets no example file of its own on
# purpose: with G fixed and a scalar outcome reward it works out to exactly `G/(G-1)`
# times this advantage (x1.14 at G=8), so it is a learning-rate change wearing a
# different name.  `--scale-rewards` does not cancel it: the divisor is the std of
# the *rewards*, not of the advantages.
#
# `--policy-loss-type` picks what gets clipped:
#
#   default     one ratio per token, `min(r*A, clip(r)*A)`.  Above ~0.2 `clip_frac`
#               means the step is too large for `--epsilon-{low,high}`.
#   gspo        one length-normalised ratio per *sequence*, the geometric mean
#               `exp(mean_t log r_t)`.  It barely strays from 1, so epsilon drops
#               two orders of magnitude (try 3e-3): a completion is in or out whole.
#   gspo-token  the same value, but the gradient is routed through each token's own
#               logp -- trust region per sequence, credit per token.
#   cispo       no `min` at all: the clipped ratio becomes a *detached* coefficient
#               on REINFORCE, so no token's gradient is ever switched off, only
#               down-weighted.  Three readings change with it: `clip_frac` now means
#               "the coefficient was clamped" and may sit near 1.0 while the loss
#               still falls; `loss` is not comparable to the default branch (it
#               grows with `-log p` instead of being bounded by |A|); and `--delta`
#               is asserted away, there being no unclipped branch left to bound.
#               Note `--epsilon-low 0` *pins* the lower bound at 1.0 and cancels all
#               down-weighting -- to open the lower side, raise it (e.g. 1.0).
#
# `--num-iterations 2` (or a `--global-batch-size` smaller than
# `--rollout-batch-size`) replays the rollout buffer, which is what makes the
# clipped ratio live at all; with one pass over one buffer, ratio == 1 identically
# and the loss degenerates to REINFORCE with a group baseline.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen3.5-4B}

python3 -m nanotrainllm.cli \
    --stage grpo \
    --model "$MODEL" \
    --data examples/data/grpo.jsonl \
    --output-dir output/grpo \
    --max-length 1024 \
    --num-generations 8 \
    --max-completion-length 512 \
    --temperature 1.0 \
    --micro-batch-size 2 \
    --grad-accum 4 \
    --lr 1e-6 \
    --reward-funcs accuracy,format \
    --log-samples-every 1 \
    --kl-coef 0.0 \
    --epochs 1
