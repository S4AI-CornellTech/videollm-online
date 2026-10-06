#!/bin/bash
# Zero-shot COIN evaluation of the released checkpoint on a single GPU.
#
# Differs from live1+_evaluate.sh in three ways:
#   - 1 GPU instead of 8 (GB10 is a single-GPU box)
#   - sdpa instead of flash_attention_2 (flash-attn is not installed / not built
#     for this arch; the README documents sdpa as the fallback)
#   - loads the released Ego4D-trained adapter rather than a COIN-finetuned one,
#     so this measures ZERO-SHOT transfer and is NOT comparable to the paper
#
# FPS selects which pre-extracted embedding directory to read; run
# `python -m data.coin.preprocess_coin --frame_fps $FPS` first.
#
# max_num_frames defaults to FPS*600, holding the streamed window at 600s of
# video regardless of frame rate. Without this, raising FPS to 10 would silently
# truncate 28.5% of test videos to their first 120s and confound the comparison.
#
# Usage:
#   bash scripts/coin/live1+_evaluate_1gpu.sh                    # 2 fps
#   FPS=10 bash scripts/coin/live1+_evaluate_1gpu.sh             # 10 fps
#   FPS=10 bash scripts/coin/live1+_evaluate_1gpu.sh coin_step_test

set -euo pipefail
cd "$(dirname "$0")/../.."

FPS="${FPS:-2}"
MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-$((FPS * 600))}"
DATASETS="${*:-coin_task_test}"

echo "fps=$FPS  max_num_frames=$MAX_NUM_FRAMES  datasets=$DATASETS"

torchrun --nproc_per_node=1 --standalone evaluate.py \
    --live_version live1+ \
    --eval_datasets $DATASETS \
    --resume_from_checkpoint chenjoya/videollm-online-8b-v1plus \
    --attn_implementation sdpa \
    --frame_fps "$FPS" \
    --embed_mark "${FPS}fps_384_1+3x3" \
    --max_num_frames "$MAX_NUM_FRAMES" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --prediction_loss_only False \
    --dataloader_num_workers 4 \
    --bf16 True \
    --tf32 True \
    --report_to none \
    --output_dir "outputs/coin_benchmarks/live1+_zeroshot_${FPS}fps/"
