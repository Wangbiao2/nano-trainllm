#!/usr/bin/env bash
# Supervised fine-tuning, including the vision rows in `sft.jsonl`.
#
# Same packed layout as pt.sh, with one difference that matters: only the
# assistant turns are supervised, so the `supervised_frac` line the run prints is
# well under 1.0 and the global-token denominator counts *those* tokens only.
# `--freeze-vit` (default) keeps the ViT frozen and trains the aligner, which is
# the usual VL recipe; add `--freeze-aligner` to train the language side alone.
#
# Multi-GPU.  There is no separate fsdp2 example: it is a launcher change plus one
# flag, and `--tuner full` needs it (one card cannot hold bf16 params + fp32
# master + two fp32 moments for a 4B model).
#
#     torchrun --nproc-per-node 8 -m nanotrainllm.cli \
#         --stage sft --model "$MODEL" --data examples/data/sft.jsonl \
#         --fsdp2 --tuner full --micro-batch-size 1 --grad-accum 8 --lr 1e-5
#
# The effective batch is `micro x accum x world_size`, and each rank must get the
# same number of units: `dataset.batches` drops the tail that does not divide by
# `world_size`, so with 8 ranks a 20-sample file trains on 16.  The denominator is
# all-reduced before any backward, so the loss is identical at any world size --
# that is the point of `denom()` living in `stages/` rather than in the loss.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen3.5-4B}

python3 -m nanotrainllm.cli \
    --stage sft \
    --model "$MODEL" \
    --data examples/data/sft.jsonl \
    --output-dir output/sft \
    --max-length 4096 \
    --packing \
    --grad-accum 8 \
    --lr 1e-5 \
    --lora-rank 16 \
    --epochs 1
