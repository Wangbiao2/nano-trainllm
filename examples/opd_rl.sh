#!/usr/bin/env bash
# OPD-RL: the teacher's log-ratio as a per-token reward, optimised by policy gradient.
#
# Same two models as opd.sh, a different place to put them.  `opd` differentiates
# the KL directly; here the per-token quantity `log p_T - log p_S` is folded into
# the GRPO advantage instead, so the update is a policy gradient whose reward
# happens to come from a teacher rather than from a rule.  That buys the clipping,
# the group baseline and the reward functions -- `--reward-funcs` still applies and
# is *added* to the teacher term, weighted by `--teacher-kl-coef`.
#
# One combination is asserted away: the teacher term is computed from the *current*
# policy's log-probs and carries no importance weight, so replaying the buffer
# would score old samples with a ratio the loss does not correct.  Keep
# `--num-iterations 1` and `--global-batch-size == --rollout-batch-size` (both
# defaults).  Everything else from grpo.sh -- `--policy-loss-type`,
# `--advantage-estimator`, `--scale-rewards` -- applies unchanged.
#
# Which of the two is better is an open question and exactly what this framework
# is for: the two stages share the rollout, the top-k machinery and the loss
# aggregation, so a comparison is one flag apart.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen3.5-4B}
TEACHER=${TEACHER:-$MODEL}

python3 -m nanotrainllm.cli \
    --stage opd-rl \
    --model "$MODEL" \
    --teacher "$TEACHER" \
    --data examples/data/grpo.jsonl \
    --output-dir output/opd-rl \
    --max-length 1024 \
    --num-generations 8 \
    --max-completion-length 512 \
    --topk 64 \
    --teacher-kl-coef 1.0 \
    --reward-funcs accuracy,format \
    --micro-batch-size 2 \
    --grad-accum 4 \
    --lr 1e-6 \
    --epochs 1
