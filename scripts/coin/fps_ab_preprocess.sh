#!/bin/bash
#SBATCH --job-name=coin-ab-prep
#SBATCH --partition=g3
#SBATCH --qos=gupta
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --exclude=g3-compute-17
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --mail-user=xc562@cornell.edu
#SBATCH --mail-type=END,FAIL
# 10 fps frames and SigLIP features for the 1000 videos of the 2 vs 10 fps experiment (datasets/coin/videos_fpstrain, from make_splits.py),
# then outputs/coin_benchmarks/splits/metadata_ab_10fps.json. Both steps skip videos already done.
# Usage (from the repo root): sbatch scripts/coin/fps_ab_preprocess.sh
set -e
cd "$SLURM_SUBMIT_DIR"
source vlo-env/bin/activate
export HF_HUB_OFFLINE=1
python -m data.preprocess.ffmpeg --frame_fps 10 --frame_resolution 384 --num_tasks 32 --video_dir datasets/coin/videos_fpstrain
python -m data.preprocess.encode --num_gpus ${SLURM_GPUS_ON_NODE:-1} --vision_pretrained google/siglip-large-patch16-384 --video_dir datasets/coin/videos_fpstrain_10fps_max384
python - <<'PY'
import os, json, torch
embed_dir = 'datasets/coin/videos_fpstrain_10fps_max384_1+3x3_google--siglip-large-patch16-384'
metadata = {f[:-3]: {'duration': (len(torch.load(os.path.join(embed_dir, f))) - 1) / 10, 'path': os.path.join(embed_dir, f)}
            for f in sorted(os.listdir(embed_dir)) if f.endswith('.pt')}
json.dump(metadata, open('outputs/coin_benchmarks/splits/metadata_ab_10fps.json', 'w'), indent=1)
print(f'10 fps features: {len(metadata)} videos')
PY
