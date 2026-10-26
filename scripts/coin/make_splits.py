# Fixed video splits for the second round of COIN fine-tuning, written to outputs/coin_benchmarks/splits/ (as COIN_METADATA files):
#   metadata_val.json         100 training videos held out for picking the best checkpoint (evaluate with the coin_*_train benchmarks)
#   metadata_train_main.json  every other training video with features, for the main 2 fps run
#   metadata_ab_2fps.json     1000 of those, for the 2 vs 10 fps experiment (the 10 fps copy is made by fps_ab_preprocess.sh)
# and datasets/coin/videos_fpstrain/ (symlinks to the 1000 raw videos, for 10 fps preprocessing).
# Usage (from the repo root): vlo-env/bin/python scripts/coin/make_splits.py
import os, json, random

out = 'outputs/coin_benchmarks/splits'
os.makedirs(out, exist_ok=True)
annos = json.load(open('datasets/coin/coin.json'))['database']
metadata = json.load(open('datasets/coin/videos_2fps_384_1+3x3_google--siglip-large-patch16-384_metadata.json'))
train = sorted(k for k in metadata if k in annos and annos[k]['subset'] == 'training')

rng = random.Random(0)
val = set(rng.sample(train, 100))
rest = [k for k in train if k not in val]
ab = sorted(rng.sample(rest, 1000))

def dump(name, keys):
    json.dump({k: metadata[k] for k in keys}, open(os.path.join(out, name), 'w'), indent=1)
    print(f'{name}: {len(keys)} training videos, {len({annos[k]["class"] for k in keys})} tasks')
dump('metadata_val.json', sorted(val))
dump('metadata_train_main.json', rest)
dump('metadata_ab_2fps.json', ab)

src = 'datasets/coin/videos_fpstrain'
os.makedirs(src, exist_ok=True)
for k in ab:
    link = os.path.join(src, f'{k}.mp4')
    if not os.path.lexists(link):
        os.symlink(os.path.abspath(f'datasets/coin/videos/{k}.mp4'), link)
