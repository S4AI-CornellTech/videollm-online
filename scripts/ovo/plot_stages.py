# Stage breakdown of the Qwen3-VL OVO-Bench runs made with --stage_timing (scripts/ovo/qwen3vl_fps_eval.py):
# (1) growth with clip length: KV cache size, vision encode, prefill, and decode per step, one colour per fps;
# (2) per fps, pies of where one question's latency goes (video decode on the CPU, vision encode, prefill, decode),
#     averaged over the questions whose clip falls in each window. Offline protocol: every question re-reads its clip.
# The first question of each file is left out (warm-up).
# Usage (Qwen3-VL env): python scripts/ovo/plot_stages.py outputs/profiling/spark_ovo_stages/qwen3vl8b_*fps.jsonl \
#     --device "DGX Spark" [--windows 0 2 5 9 13]
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

KV_BYTES_PER_TOKEN = 2 * 36 * 8 * 128 * 2 # Qwen3-VL-8B: K,V x 36 layers x 8 KV heads x 128 dims x bf16
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100'] # default palette slots 1-4 (dataviz skill)
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
STAGES = [('load_s', 'video decode (CPU)'), ('vision_s', 'vision encode'), ('prefill_s', 'prefill'), ('decode_s', 'decode')]

def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return [r for r in rows[1:] if r.get('response') is not None and 'prefill_s' in r] # rows[0]: warm-up

def growth(runs, device, out):
    plt.rcParams.update({'font.size': 11, 'figure.facecolor': SURF, 'axes.facecolor': SURF})
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f'Qwen3-VL-8B on OVO-Bench ({device}): cost of one question vs clip length', color=INK, fontsize=14,
                 fontweight='bold', x=0.01, ha='left')
    panels = [('KV cache (GB, computed)', lambda r: r['input_tokens'] * KV_BYTES_PER_TOKEN / 1e9),
              ('vision encode (s)', lambda r: r['vision_s']),
              ('prefill (s)', lambda r: r['prefill_s']),
              ('decode (ms per step)', lambda r: 1000 * r['decode_s'] / max(r['decode_steps'], 1))]
    for a, (name, f) in zip(ax.flat, panels):
        for (fps, rows), c in zip(runs, SERIES):
            a.scatter([r['realtime'] / 60 for r in rows], [f(r) for r in rows], color=c, s=16, alpha=0.7, lw=0, label=f'{fps:g} fps')
        a.set(xlabel='video up to the question (min)', title=name)
        a.grid(True, color=GRID, lw=0.8); a.set_axisbelow(True); a.spines[['top', 'right']].set_visible(False)
        a.set_xlim(left=0); a.set_ylim(bottom=0)
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc='upper left', bbox_to_anchor=(0.01, 0.955), ncol=len(runs),
               frameon=False, markerscale=1.8)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out, dpi=150); plt.close(fig)

def pies(fps, rows, windows, device, out):
    plt.rcParams.update({'font.size': 20, 'figure.facecolor': SURF, 'axes.facecolor': SURF})
    cells = []
    for lo, hi in zip(windows[:-1], windows[1:]):
        sel = [r for r in rows if lo * 60 <= r['realtime'] < hi * 60]
        if sel:
            cells.append((lo, hi, sel, np.array([np.mean([r[k] for r in sel]) for k, _ in STAGES])))
    ncol = 2; nrow = -(-len(cells) // ncol)
    fig = plt.figure(figsize=(24, 2.8 + 6.2 * nrow))
    head = 2.8 / (2.8 + 6.2 * nrow); cell_h = (1 - head) / nrow
    fig.text(0.01, 1 - 0.25 * head, f'Latency of one question — Qwen3-VL-8B, {device}, {fps:g} fps', color=INK, fontsize=26,
             fontweight='bold', va='top')
    fig.text(0.01, 1 - 0.55 * head, 'Offline OVO-Bench protocol: each question decodes and prefills its whole clip, then answers '
             '(one letter, so decode is a single step).\nMean over the questions whose clip falls in each window.',
             color=MUTED, fontsize=19, va='top')
    for k, (lo, hi, sel, s) in enumerate(cells): # pie on the left, its numbers on the right
        x0, y0 = (k % ncol) / ncol, 1 - head - (k // ncol + 1) * cell_h
        total = s.sum()
        a = fig.add_axes([x0 + 0.01, y0 + 0.04 * cell_h, 0.2, 0.8 * cell_h])
        wedges, _ = a.pie(s, colors=SERIES, startangle=90, counterclock=False, wedgeprops=dict(edgecolor=SURF, linewidth=3))
        big = int(np.argmax(s))
        ang = np.deg2rad((wedges[big].theta1 + wedges[big].theta2) / 2)
        a.text(0.55 * np.cos(ang), 0.55 * np.sin(ang), f'{100 * s[big] / total:.2f}%', ha='center', va='center', color='white',
               fontsize=22, fontweight='bold')
        a.set_aspect('equal')
        t = fig.add_axes([x0 + 0.23, y0 + 0.04 * cell_h, 0.26, 0.8 * cell_h]); t.axis('off')
        t.text(0, 0.95, f'clips {lo:g}–{hi:g} min', color=INK, fontsize=22, fontweight='bold', va='top', transform=t.transAxes)
        for j, ((_, n), v, c) in enumerate(zip(STAGES, s, SERIES)):
            yy = 0.74 - 0.12 * j
            t.add_patch(plt.Rectangle((0, yy - 0.032), 0.045, 0.064, color=c, transform=t.transAxes, clip_on=False))
            t.text(0.07, yy, n, color=INK, va='center', transform=t.transAxes)
            t.text(0.80, yy, f'{v:.2f} s', color=INK, va='center', ha='right', transform=t.transAxes)
            t.text(1.05, yy, f'{100 * v / total:.2f}%', color=INK, va='center', ha='right', transform=t.transAxes)
        t.text(0, 0.22, f'total {total:.2f} s · {len(sel)} questions\n~{np.mean([r["input_tokens"] for r in sel]) / 1000:.1f}k tokens '
               f'({np.mean([r["input_tokens"] for r in sel]) * KV_BYTES_PER_TOKEN / 1e9:.1f} GB KV)', color=MUTED, fontsize=18,
               va='top', linespacing=1.4, transform=t.transAxes)
    fig.savefig(out, dpi=150, bbox_inches='tight', pad_inches=0.2); plt.close(fig)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('files', nargs='+')
    p.add_argument('--device', default='DGX Spark')
    p.add_argument('--windows', type=float, nargs='+', default=[0, 2, 5, 9, 13], help='clip-length window edges (min)')
    p.add_argument('--out_dir', default=None)
    a = p.parse_args()
    runs = sorted(((json.loads(open(f).readline())['fps'], load(f)) for f in a.files), key=lambda x: x[0])
    out_dir = a.out_dir or os.path.dirname(a.files[0])
    growth(runs, a.device, os.path.join(out_dir, 'stages_growth.png'))
    for fps, rows in runs:
        pies(fps, rows, a.windows, a.device, os.path.join(out_dir, f'stages_pies_{fps:g}fps.png'))
        s = {k: np.mean([r[k] for r in rows]) for k, _ in STAGES}
        print(f'{fps:g} fps ({len(rows)} questions): ' + ', '.join(f'{n} {s[k]:.2f} s' for k, n in STAGES))
    print('->', out_dir)

if __name__ == '__main__':
    main()
