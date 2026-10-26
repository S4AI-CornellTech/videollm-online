#!/bin/bash
#SBATCH --job-name=coin-ft-v1plus
#SBATCH --partition=g3
#SBATCH --qos=gupta
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --mail-user=xc562@cornell.edu
#SBATCH --mail-type=BEGIN,FAIL,REQUEUE
# Step 3/3: continue training the released VideoLLM-online-8B-v1+ LoRA on the five COIN benchmarks.
# Same recipe as live1+.sh except --init_from_checkpoint and no end-of-training eval (run live1+_from_v1plus_evaluate.sh on the final test set); sdpa because flash-attn has no B300 (sm_103) wheel.
cd "$SLURM_SUBMIT_DIR"
source vlo-env/bin/activate
NGPU=${SLURM_GPUS_ON_NODE:-8}
# keep the effective batch at 64 (= GPUs x accumulation) whatever --gres was requested
torchrun --nproc_per_node=$NGPU --standalone train.py --deepspeed configs/deepspeed/zero1.json \
    --live_version live1+ \
    --init_from_checkpoint chenjoya/videollm-online-8b-v1plus \
    --attn_implementation sdpa \
    --train_datasets coin_step_train coin_next_train coin_task_train coin_procedure_train coin_taskprocedure_train \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $((64 / NGPU)) \
    --gradient_checkpointing True \
    --eval_strategy no \
    --prediction_loss_only False \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 2 \
    --learning_rate 0.0001 \
    --optim adamw_torch \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --logging_steps 10 \
    --dataloader_num_workers 16 \
    --bf16 True \
    --tf32 True \
    --report_to tensorboard \
    --output_dir outputs/coin_benchmarks/live1+_from_v1plus
