#!/bin/bash
# Step 2/3: preprocess COIN videos in two jobs, so neither waits for a free 8-GPU node:
#   a) ffmpeg 2fps/384 frames on CPUs only   b) SigLIP CLS + 3x3 pooled tokens on 1 GPU (after a)
# Both skip videos already done, so just re-run this after uploading more videos.
# Usage (from the repo root): bash scripts/coin/slurm_preprocess.sh   (AFTER=<jobid> to encode only after another encode job; EXCLUDE=<nodes> to avoid broken GPU nodes)
set -e
cd "$(dirname "$0")/../.."
mkdir -p outputs/slurm
FFMPEG=$(sbatch --parsable --mail-user=xc562@cornell.edu --mail-type=BEGIN,FAIL -p g3 --qos=gupta -J coin-ffmpeg --cpus-per-task=120 --mem=128G --time=12:00:00 -o outputs/slurm/%x-%j.out \
    --wrap 'vlo-env/bin/python -m data.preprocess.ffmpeg --frame_fps 2 --frame_resolution 384 --num_tasks 12 --video_dir datasets/coin/videos' --chdir "$PWD")
ENCODE=$(sbatch --parsable --mail-user=xc562@cornell.edu --mail-type=BEGIN,FAIL -p g3 --qos=gupta -J coin-encode --gres=gpu:${NGPU:-1} ${EXCLUDE:+--exclude=$EXCLUDE} --cpus-per-task=32 --mem=256G --time=12:00:00 -o outputs/slurm/%x-%j.out \
    --dependency=afterok:$FFMPEG${AFTER:+,afterany:$AFTER} --chdir "$PWD" scripts/coin/slurm_encode.sh)
echo "ffmpeg job $FFMPEG -> encode job $ENCODE"
echo $ENCODE > outputs/slurm/last_encode_job
