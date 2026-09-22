#!/usr/bin/env bash
# DPO: no reward model, no sampling -- four sequence log-probabilities.
#
# Two flags here are not free choices.  `--micro-batch-size 2` is the minimum,
# because a preference pair must live inside one forward pass: `stages/dpo.py`
# recovers the sides with `[0::2]` / `[1::2]` and asserts an even row count.  And
# there is deliberately no `--packing` -- packing merges samples into one row, and
# DPO needs one row per side to sum its log-probs separately.
#
# `--tuner lora` also means no reference model is loaded: the frozen base weights
# under the adapter *are* `pi_ref`, so this run holds one copy of the model
# instead of two.  Under `--tuner full` a real reference is built.
#
# `--dpo-beta` is the KL strength, not a learning rate: the gradient carries
# `sigmoid(-beta * margin)`, so it anneals itself once a pair is ranked right.
# Two variants need no second script: `--dpo-reference-free` drops the `pi_ref`
# term (CPO's objective -- cheaper, and worse, because nothing then stops both
# log-probs from falling together), and `--dpo-sft-weight 1.0` adds an NLL term on
# the chosen response (RPO), which anchors what DPO alone leaves free.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen3.5-4B}

python3 -m nanotrainllm.cli \
    --stage dpo \
    --model "$MODEL" \
    --data examples/data/dpo.jsonl \
    --output-dir output/dpo \
    --max-length 2048 \
    --micro-batch-size 2 \
    --grad-accum 4 \
    --lr 5e-6 \
    --dpo-beta 0.1 \
    --epochs 1
