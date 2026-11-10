# Growth of KV cache size, peak GPU memory and latency with clip length for the Qwen3-VL OVO-Bench fps runs
# (scripts/ovo/qwen3vl_fps_eval.py). Offline protocol: every question prefills its whole clip from scratch.
# KV size is computed (input tokens x 144 KiB: 36 layers x 8 KV heads x 128 dims x K,V x bf16), not measured.
# generate_s = vision encode + prefill + decode (answers are 1-2 tokens, so it is almost all encode + prefill).
# Usage (Qwen3-VL env): python scripts/ovo/plot_growth.py outputs/ovo/qwen3vl8b_{0.5,1,2}fps.jsonl --out outputs/ovo/growth.png
import argparse, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

KV_BYTES_PER_TOKEN = 2 * 36 * 8 * 128 * 2
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300'] # default palette slots 1-6 (dataviz skill), one per fps
INK, INK2, GRID, SURFACE = '#1f1f1e', '#5f5e58', '#e6e5e0', '#fcfcfb'

def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return [r for r in rows if r.get('response') is not None and r.get('input_tokens')]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('files', nargs='+')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='1× B300', help='named in the title')
    args = p.parse_args()
    by_fps = {} # files with the same fps (e.g. a smoke test) are merged
    for f in args.files:
        rows = load(f)
        by_fps.setdefault(rows[0]['fps'], []).extend(rows)
    runs = []
    for fps, rows in by_fps.items():
        a = lambda k: np.array([r[k] for r in rows], dtype=float)
        runs.append(dict(fps=fps, t=np.minimum(a('realtime'), 1e9), tok=a('input_tokens'), gen=a('generate_s'),
                         load=a('load_s'), peak=a('peak_gb'), n=len(rows)))
    runs.sort(key=lambda r: r['fps'])

    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                         'ytick.color': INK2, 'axes.titlecolor': INK, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE})
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.suptitle(f'Qwen3-VL-8B on OVO-Bench (offline, {args.device}): growth with clip length', color=INK, fontsize=13, x=0.01, ha='left')
    style = dict(s=9, alpha=0.55, linewidths=0)

    # 1. KV cache size vs clip time, per fps
    a = ax[0, 0]
    for r, c in zip(runs, SERIES):
        a.scatter(r['t'] / 60, r['tok'] * KV_BYTES_PER_TOKEN / 1e9, color=c, label=f'{r["fps"]:g} fps (n={r["n"]})', **style)
    a.set(xlabel='video up to the question (min)', ylabel='KV cache (GB, computed)', title='KV cache size')
    # 2. peak GPU memory vs input tokens
    a = ax[0, 1]
    for r, c in zip(runs, SERIES):
        a.scatter(r['tok'] / 1e3, r['peak'], color=c, **style)
    allt = np.concatenate([r['tok'] for r in runs]); allp = np.concatenate([r['peak'] for r in runs])
    k, b = np.polyfit(allt, allp, 1)
    xs = np.linspace(0, allt.max(), 50)
    a.plot(xs / 1e3, k * xs + b, color=INK2, lw=1.5, ls='--')
    a.text(0.03, 0.95, f'fit: {b:.1f} GB + {k * 1e3:.3f} GB per 1k tokens', transform=a.transAxes, va='top', color=INK2)
    a.set(xlabel='input tokens (k)', ylabel='peak GPU memory (GB, measured)', title='Peak GPU memory')
    # 3. encode + prefill (+ tiny decode) latency vs input tokens, quadratic fit
    a = ax[1, 0]
    for r, c in zip(runs, SERIES):
        a.scatter(r['tok'] / 1e3, r['gen'], color=c, **style)
    allg = np.concatenate([r['gen'] for r in runs])
    q = np.polyfit(allt / 1e3, allg, 2)
    a.plot(xs / 1e3, np.polyval(q, xs / 1e3), color=INK2, lw=1.5, ls='--')
    a.text(0.03, 0.95, f'fit: {q[2]:.2f} + {q[1]:.3f}·T + {q[0] * 1e3:.3f}e-3·T² s  (T = k tokens)', transform=a.transAxes, va='top', color=INK2)
    a.set(xlabel='input tokens (k)', ylabel='seconds', title='Vision encode + prefill + decode per question')
    # 4. video loading (decode + resize on CPU) vs frames
    a = ax[1, 1]
    for r, c in zip(runs, SERIES):
        a.scatter(r['t'] / 60, r['load'], color=c, **style)
    a.set(xlabel='video up to the question (min)', ylabel='seconds', title='Video decode + resize per question (CPU)')
    for a in ax.flat:
        a.grid(True, color=GRID, lw=0.8); a.set_axisbelow(True)
        a.spines[['top', 'right']].set_visible(False)
        a.set_xlim(left=0); a.set_ylim(bottom=0)
    fig.legend(*ax[0, 0].get_legend_handles_labels(), frameon=False, loc='upper left', bbox_to_anchor=(0.01, 0.955),
               ncol=len(runs), markerscale=2, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(args.out, dpi=130)

    print(f'peak GB = {b:.2f} + {k * 1e3:.4f} per 1k tokens | latency fit (T in k tokens): {q[2]:.3f} + {q[1]:.4f} T + {q[0]:.6f} T^2 s')
    for r in runs:
        i = np.argsort(r['tok'])
        print(f'{r["fps"]:g} fps: n={r["n"]}, tokens median {np.median(r["tok"]):.0f} max {r["tok"].max():.0f}, '
              f'KV max {r["tok"].max() * KV_BYTES_PER_TOKEN / 1e9:.1f} GB, gen median {np.median(r["gen"]):.2f} s max {r["gen"].max():.1f} s, '
              f'load median {np.median(r["load"]):.2f} s max {r["load"].max():.1f} s, peak max {r["peak"].max():.1f} GB, '
              f'tokens per video-second {np.median(r["tok"] / np.maximum(r["t"], 1)):.1f}')

if __name__ == '__main__':
    main()
