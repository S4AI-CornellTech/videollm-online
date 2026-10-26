#!/bin/bash
#SBATCH --job-name=coin-eval
#SBATCH --partition=g3
#SBATCH --qos=gupta
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --mail-user=xc562@cornell.edu
#SBATCH --mail-type=BEGIN,FAIL
# CKPT=chenjoya/videollm-online-8b-v1plus gives the zero-shot baseline before fine-tuning.
cd "$SLURM_SUBMIT_DIR"
source vlo-env/bin/activate
CKPT=${CKPT:-outputs/coin_benchmarks/live1+_from_v1plus}
torchrun --nproc_per_node=${SLURM_GPUS_ON_NODE:-8} --standalone evaluate.py \
    --live_version live1+ \
    --attn_implementation sdpa \
    --eval_datasets coin_step_test coin_next_test coin_task_test coin_procedure_test coin_taskprocedure_test \
    --per_device_eval_batch_size 1 \
    --prediction_loss_only False \
    --dataloader_num_workers 16 \
    --bf16 True \
    --tf32 True \
    --report_to tensorboard \
    --output_dir outputs/coin_benchmarks/eval/$(basename $CKPT) \
    --resume_from_checkpoint $CKPT
