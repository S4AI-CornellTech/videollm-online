#!/bin/bash
#SBATCH --job-name=coin-fps
#SBATCH --partition=g3
#SBATCH --qos=gupta
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --mail-user=xc562@cornell.edu
#SBATCH --mail-type=END,FAIL
# Quick test-time FPS check: the fine-tuned model (trained at 2 fps) evaluated at FPS frames per second
# on the same 150 random test videos (datasets/coin/videos_fpstest, symlinks), step + task benchmarks only.
# Step clips are short (median 7 s) so they fit Llama-3's 8k context at 10 fps; most task (whole video) inputs do not.
# Usage (from the repo root), one job per fps so they run in parallel on separate GPUs:
#   FPS=2 sbatch scripts/coin/fps_compare.sh;  FPS=10 sbatch scripts/coin/fps_compare.sh;  FPS=0.5 (fractional fps works too)
# ROPE=4: also stretch the LLM context with RoPE scaling (results go to eval_<fps>fps_rope<ROPE>); ROPE_TYPE=yarn for static
#   YaRN instead of the default dynamic NTK (results go to eval_<fps>fps_rope<ROPE>yarn)
# CKPT=<adapter dir> TAG=<suffix for the result name> EVAL_DATASETS="<benchmarks>": evaluate another checkpoint / other benchmarks
set -e
cd "$SLURM_SUBMIT_DIR"
source vlo-env/bin/activate
export HF_HUB_OFFLINE=1
FPS=${FPS:?set FPS}
CKPT=${CKPT:-outputs/coin_benchmarks/live1+_from_v1plus}
OUT=outputs/coin_benchmarks/fps_test # metadata_2fps.json there is the 150-video subset of the 2 fps snapshot
NAME=eval_${FPS}fps${ROPE:+_rope$ROPE}${ROPE_TYPE:+$ROPE_TYPE}${TAG}

if [ ! -f $OUT/metadata_${FPS}fps.json ]; then
    # frames and SigLIP features at this fps for the 150 videos (both steps skip videos already done)
    python -m data.preprocess.ffmpeg --frame_fps $FPS --frame_resolution 384 --num_tasks 32 --video_dir datasets/coin/videos_fpstest
    python -m data.preprocess.encode --num_gpus 1 --vision_pretrained google/siglip-large-patch16-384 --video_dir datasets/coin/videos_fpstest_${FPS}fps_max384
    python - $OUT $FPS <<'PY'
import os, sys, json, torch
out, fps = sys.argv[1], float(sys.argv[2])
embed_dir = f'datasets/coin/videos_fpstest_{fps:g}fps_max384_1+3x3_google--siglip-large-patch16-384'
metadata = {f[:-3]: {'duration': (len(torch.load(os.path.join(embed_dir, f))) - 1) / fps, 'path': os.path.join(embed_dir, f)}
            for f in sorted(os.listdir(embed_dir)) if f.endswith('.pt')}
json.dump(metadata, open(os.path.join(out, f'metadata_{fps:g}fps.json'), 'w'), indent=1)
print(f'{fps:g} fps features: {len(metadata)} videos')
PY
fi

echo "===== evaluating at $FPS fps${ROPE:+, RoPE scaling x$ROPE ${ROPE_TYPE:-dynamic}}"
# max_num_frames high enough that no question loses its video at either fps
COIN_METADATA=$OUT/metadata_${FPS}fps.json torchrun --nproc_per_node=1 --standalone evaluate.py \
    --live_version live1+ \
    --attn_implementation sdpa \
    --frame_fps $FPS \
    --embed_mark ${FPS}fps_384_1+3x3 \
    --max_num_frames 6000 \
    ${ROPE:+--rope_scaling_factor $ROPE} \
    ${ROPE_TYPE:+--rope_scaling_type $ROPE_TYPE} \
    --eval_datasets ${EVAL_DATASETS:-coin_step_test coin_task_test} \
    --per_device_eval_batch_size 1 \
    --prediction_loss_only False \
    --dataloader_num_workers 16 \
    --bf16 True \
    --tf32 True \
    --report_to none \
    --output_dir $OUT/$NAME \
    --resume_from_checkpoint "$CKPT" > $OUT/$NAME.log 2>&1 \
    || { echo "!!! $NAME failed, end of $OUT/$NAME.log:"; tail -30 $OUT/$NAME.log; exit 1; }
tr '\r' '\n' < $OUT/$NAME.log | grep -oE "'eval_coin_[a-z]+_test_(accuracy|runtime)': [0-9.]+"
