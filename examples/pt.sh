#!/usr/bin/env bash
# Continued pre-training: one loss on every token, no template, no roles.
#
# `--packing` is the whole point of this stage.  A pre-training corpus is a pile
# of documents of wildly different lengths; packed into 4096-token bins there is
# no padding at all, and since every token is supervised the throughput gain is
# the full one.  Packing implies padding-free, which needs a varlen
# flash-attention kernel -- hence `--attn-impl flash_attention_2` (the default),
# and hence `--packing` not being reachable from `--dry-run`, which is CPU/sdpa.
#
# `--packing` also pins `--micro-batch-size 1` (asserted): a packed micro-batch is
# exactly one `max_length` bin, so scale the step with `--grad-accum` instead.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen3.5-4B}

python3 -m nanotrainllm.cli \
    --stage pt \
    --model "$MODEL" \
    --data examples/data/pt.jsonl \
    --output-dir output/pt \
    --max-length 4096 \
    --packing \
    --grad-accum 8 \
    --lr 1e-5 \
    --epochs 1
