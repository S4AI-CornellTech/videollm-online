#!/bin/bash
#SBATCH --job-name=ovo-qwen3vl
#SBATCH --partition=g3
#SBATCH --qos=gupta
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --mail-user=xc562@cornell.edu
#SBATCH --mail-type=END,FAIL
# OVO-Bench backward + real-time tasks with Qwen3-VL at FPS frames per second (scripts/ovo/qwen3vl_fps_eval.py).
# The Qwen3-VL env, OVO-Bench data and weights live on /share (home quota is small): see OVO / HF_HOME below.
# Usage (from the videollm-online root): FPS=1 sbatch scripts/ovo/qwen3vl_fps.sh   [MODEL=Qwen/Qwen3-VL-4B-Instruct TAG=4b]
#   EXTRA="--limit 20" for a smoke test. Rerunning resumes from the JSONL.
set -e
cd "$SLURM_SUBMIT_DIR"
OVO=/share/gupta/xc562/ovo
export HF_HOME=/share/gupta/xc562/hf_cache HF_HUB_OFFLINE=1
FPS=${FPS:?set FPS}
MODEL=${MODEL:-Qwen/Qwen3-VL-8B-Instruct}
TAG=${TAG:-8b}
OUT=outputs/ovo/qwen3vl${TAG}_${FPS}fps${SUFFIX}.jsonl
$OVO/qwen-env/bin/python scripts/ovo/qwen3vl_fps_eval.py --fps $FPS --model_path $MODEL --out $OUT \
    --anno $OVO/OVO-Bench/data/ovo_bench_new.json --video_dir ${VIDEO_DIR:-$OVO/OVO-Bench/data/src_videos} $EXTRA
$OVO/qwen-env/bin/python scripts/ovo/score_fps.py $OUT
