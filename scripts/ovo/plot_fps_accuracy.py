# Accuracy vs fps for the Qwen3-VL OVO-Bench runs (scripts/ovo/qwen3vl_fps_eval.py), on the questions every run answered
# (high fps skips clips over the 262k context), with paired-bootstrap 95% intervals, next to GPU time per question.
# Usage (Qwen3-VL env): python scripts/ovo/plot_fps_accuracy.py outputs/ovo/qwen3vl8b_{0.5,1,2,4,8,16}fps.jsonl --out outputs/ovo/fps_accuracy.png
import argparse, json, random
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BACKWARD = ['EPM', 'ASI', 'HLD']
REALTIME = ['OCR', 'ACR', 'ATR', 'STU', 'FPD', 'OJR']
SERIES = ['#2a78d6', '#eb6834', '#1baf7a'] # default palette slots 1-3 (dataviz skill)
INK, INK2, GRID, SURFACE = '#1f1f1e', '#5f5e58', '#e6e5e0', '#fcfcfb'

def correct(r):
    return int(r.get('response') is not None and r['ground_truth'] in r['response'])

def scores(rows, ids):
    by = defaultdict(list)
    for i in ids:
        by[rows[i]['task']].append(correct(rows[i]))
    cat = lambda T: 100 * np.mean([np.mean(by[t]) for t in T if by[t]])
    b, r = cat(BACKWARD), cat(REALTIME)
    return dict(overall=(b + r) / 2, backward=b, realtime=r)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('files', nargs='+')
    p.add_argument('--out', required=True)
    p.add_argument('--boot', type=int, default=1000)
    args = p.parse_args()
    runs = {}
    for f in args.files:
        rows = {r['id']: r for r in map(json.loads, open(f))}
        runs[next(iter(rows.values()))['fps']] = rows
    fps = sorted(runs)
    ids = [i for i in runs[fps[0]] if all(i in runs[f] and not runs[f][i].get('skipped') for f in fps)]
    random.seed(0)
    boots = [[random.choice(ids) for _ in ids] for _ in range(args.boot)]
    keys = ['overall', 'realtime', 'backward']
    point = {f: scores(runs[f], ids) for f in fps}
    lo, hi = {}, {}
    for f in fps:
        s = {k: sorted(scores(runs[f], b)[k] for b in boots) for k in keys}
        lo[f] = {k: s[k][int(0.025 * args.boot)] for k in keys}; hi[f] = {k: s[k][int(0.975 * args.boot)] for k in keys}
    gen = {f: np.mean([runs[f][i]['generate_s'] for i in ids]) for f in fps}

    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                         'ytick.color': INK2, 'axes.titlecolor': INK, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    fig.suptitle(f'Qwen3-VL-8B on OVO-Bench backward + real-time: the {len(ids)} questions every fps fits in context',
                 color=INK, fontsize=12, x=0.01, ha='left')
    a = ax[0]
    names = {'overall': 'Overall (B+R)/2', 'realtime': 'Real-time avg', 'backward': 'Backward avg'}
    for k, c in zip(keys, SERIES):
        y = [point[f][k] for f in fps]
        a.fill_between(fps, [lo[f][k] for f in fps], [hi[f][k] for f in fps], color=c, alpha=0.12, lw=0)
        a.plot(fps, y, color=c, lw=2, marker='o', ms=8, mec=SURFACE, mew=2)
        a.text(fps[-1] * 1.12, y[-1], names[k], color=INK, va='center')
    a.set(xscale='log', xlabel='frames per second (log scale)', ylabel='accuracy (%)', title='Accuracy (bands: 95% bootstrap)')
    a.set_xlim(fps[0] / 1.3, fps[-1] * 3.2)
    a = ax[1]
    a.plot(fps, [gen[f] for f in fps], color=SERIES[0], lw=2, marker='o', ms=8, mec=SURFACE, mew=2)
    for f in fps:
        a.annotate(f'{gen[f]:.2g} s', (f, gen[f]), textcoords='offset points', xytext=(0, 9), ha='center', color=INK2)
    a.set(xscale='log', yscale='log', xlabel='frames per second (log scale)', ylabel='seconds (log scale)',
          title='Mean GPU time per question (1× B300)')
    a.set_ylim(top=max(gen.values()) * 2.5)
    for a in ax:
        a.set_xticks(fps); a.set_xticklabels([f'{f:g}' for f in fps]); a.minorticks_off()
        a.grid(True, color=GRID, lw=0.8); a.set_axisbelow(True); a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(args.out, dpi=130)
    for f in fps:
        print(f'{f:g} fps: ' + ', '.join(f'{k} {point[f][k]:.1f} [{lo[f][k]:.1f}, {hi[f][k]:.1f}]' for k in keys) + f', GPU {gen[f]:.2f} s')

if __name__ == '__main__':
    main()
