#!/bin/bash
#SBATCH --job-name=coin-train
#SBATCH --partition=g3
#SBATCH --qos=gupta
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --exclude=g3-compute-17
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --mail-user=xc562@cornell.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
# Continue training the released VideoLLM-online-8B-v1+ LoRA on the five COIN benchmarks, same recipe as live1+_from_v1plus.sh,
# on the videos listed in METADATA (a COIN_METADATA file from make_splits.py), at FPS frames per second.
# Resumes from the latest checkpoint in OUT, so a job stopped by the 24 h limit can just be resubmitted.
# Usage (from the repo root):
#   OUT=outputs/coin_benchmarks/<name> METADATA=outputs/coin_benchmarks/splits/<file>.json [EPOCHS=2] [FPS=2] [ROPE=4] [MAX_FRAMES=1200] [SAVE_STEPS=300] \
#     sbatch -J <job name> scripts/coin/train_coin.sh
set -e
cd "$SLURM_SUBMIT_DIR"
source vlo-env/bin/activate
export HF_HUB_OFFLINE=1
export COIN_METADATA=${METADATA:?set METADATA}
FPS=${FPS:-2}
NGPU=${SLURM_GPUS_ON_NODE:-1}
echo "training on $(python -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" $COIN_METADATA) videos at $FPS fps${ROPE:+, RoPE scaling x$ROPE} -> ${OUT:?set OUT}"
# keep the effective batch at 64 (= GPUs x accumulation) whatever --gres was requested
torchrun --nproc_per_node=$NGPU --standalone train.py --deepspeed configs/deepspeed/zero1.json \
    --live_version live1+ \
    --init_from_checkpoint chenjoya/videollm-online-8b-v1plus \
    --attn_implementation sdpa \
    --frame_fps $FPS \
    --embed_mark ${FPS}fps_384_1+3x3 \
    --max_num_frames ${MAX_FRAMES:-1200} \
    ${ROPE:+--rope_scaling_factor $ROPE} \
    --train_datasets coin_step_train coin_next_train coin_task_train coin_procedure_train coin_taskprocedure_train \
    --num_train_epochs ${EPOCHS:-2} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $((64 / NGPU)) \
    --gradient_checkpointing True \
    --eval_strategy no \
    --prediction_loss_only False \
    --save_strategy steps \
    --save_steps ${SAVE_STEPS:-300} \
    --learning_rate 0.0001 \
    --optim adamw_torch \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --logging_steps 10 \
    --dataloader_num_workers 16 \
    --bf16 True \
    --tf32 True \
    --report_to tensorboard \
    --output_dir $OUT
