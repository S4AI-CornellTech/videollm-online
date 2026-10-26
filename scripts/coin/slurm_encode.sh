#!/bin/bash
# Submitted by slurm_preprocess.sh: SigLIP features on however many GPUs were allocated.
set -e
source vlo-env/bin/activate
python -m data.preprocess.encode --num_gpus ${SLURM_GPUS_ON_NODE:-1} --vision_pretrained google/siglip-large-patch16-384 --video_dir datasets/coin/videos_2fps_max384
# encode writes videos_2fps_max384_1+3x3_*, but data/coin/coin.py reads videos_{embed_mark}_* = videos_2fps_384_1+3x3_*
ln -sfn videos_2fps_max384_1+3x3_google--siglip-large-patch16-384 datasets/coin/videos_2fps_384_1+3x3_google--siglip-large-patch16-384
# rebuild the metadata data/coin/coin.py caches, once and atomically here, so stale metadata never hides new videos
# and parallel training ranks never race to write it
python - <<'PY'
import os, json, torch
embed_dir = 'datasets/coin/videos_2fps_384_1+3x3_google--siglip-large-patch16-384'
metadata = {}
for file in sorted(os.listdir(embed_dir)):
    if file.endswith('.pt'):
        path = os.path.join(embed_dir, file)
        metadata[file[:-3]] = {'duration': (len(torch.load(path)) - 1) / 2, 'path': path} # 2 fps, same as COIN.get_metadata
json.dump(metadata, open(embed_dir + '_metadata.json.tmp', 'w'), indent=4)
os.replace(embed_dir + '_metadata.json.tmp', embed_dir + '_metadata.json')
print(f'encoded: {len(metadata)} videos')
PY
