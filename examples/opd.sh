#!/usr/bin/env bash
# On-policy distillation: the KL to the teacher *is* the loss.
#
# The name is the whole idea.  Classic sequence-level KD trains on text the
# teacher wrote, so the student only ever sees states it would not have visited;
# `--on-policy-ratio 1.0` makes the *student* write the text and asks the teacher
# to score it, which puts the KL exactly where the student's errors are.  Set it
# to 0 and this becomes ordinary off-policy KD on the dataset's own responses -- a
# useful A/B, and the reason the flag is a ratio rather than a boolean (it is one
# coin flip per micro-batch, which is GKD's `lmbda`).
#
# `--union-topk` (default) gathers both models at the union of their top-`--topk`
# indices.  Teacher-only top-k, reachable with `--no-union-topk`, silently drops
# every token the student ranks highly and the teacher does not -- which is
# precisely the mass reverse KL is supposed to punish.  `--with-tail` adds one
# logsumexp("everything else") bucket so the truncated distribution still sums to 1,
# making the result a lower bound rather than an approximation of unknown sign.
#
# `--jsd-beta 1.0` is reverse KL, `KL(student||teacher)`: mode-seeking, "do not say
# things the teacher would not say".  `0.0` is the forward KL, mass-covering, so the
# student hedges; `0.5` is plain Jensen-Shannon.
#
# TEACHER defaults to the student for a plumbing check.  A real run wants a bigger
# checkpoint from the same tokenizer family -- the top-k indices are compared as
# integers, so the two vocabularies must be identical (asserted at construction).
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen3.5-4B}
TEACHER=${TEACHER:-$MODEL}

python3 -m nanotrainllm.cli \
    --stage opd \
    --model "$MODEL" \
    --teacher "$TEACHER" \
    --data examples/data/sft.jsonl \
    --output-dir output/opd \
    --max-length 1024 \
    --max-completion-length 512 \
    --on-policy-ratio 1.0 \
    --topk 64 \
    --jsd-beta 1.0 \
    --micro-batch-size 2 \
    --grad-accum 4 \
    --lr 1e-5 \
    --epochs 1
