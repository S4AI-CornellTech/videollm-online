# Pie charts of the latency of one answering frame (vision encode + frame prefill + decode of its answer) at several points
# of a long run, from the <video>_growth.json and <video>_phase_events.json that demo.cli writes with --phase_timing True.
# Each pie averages over a window of video around a mark (replies come every 1-2 min, so a window has to be a few minutes
# wide to contain any): vision and prefill are per-frame means; decode = time per generated token x mean answer length.
# The first frame is left out: it includes warm-up (~20x a normal frame).
# Usage: python -m profiling_scripts.plot_breakdown outputs/profiling/spark/coin_runA_growth.json --label "run A" [--marks 0 15 30 45] [--window 5]
import argparse, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SERIES = ['#2a78d6', '#eb6834', '#1baf7a'] # default palette slots 1-3 (dataviz skill): vision, prefill, decode
INK, MUTED, SURF = '#0b0b0b', '#52514e', '#fcfcfb'

def main():
    p = argparse.ArgumentParser()
    p.add_argument('growth_json')
    p.add_argument('--label', default='')
    p.add_argument('--marks', type=float, nargs='+', default=[0, 15, 30, 45], help='minutes')
    p.add_argument('--window', type=float, default=5, help='minutes of video summed per pie')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    events = json.load(open(args.growth_json))
    pre = [e for e in events if e['phase'] == 'frame_prefill']
    dec = [e for e in events if e['phase'] == 'decode' and e['units'] > 0]
    ph = json.load(open(args.growth_json.replace('_growth.json', '_phase_events.json')))
    vis = [t1 - t0 for name, t0, t1, n in ph if name == 'vision_embed']
    assert len(vis) == len(pre), (len(vis), len(pre)) # one vision_embed per frame, in order
    t_frame = np.array([e['video_time'] for e in pre]) / 60
    s_vis, s_pre = np.array(vis), np.array([e['seconds'] for e in pre])
    t_dec, s_dec = np.array([e['video_time'] for e in dec]) / 60, np.array([e['seconds'] for e in dec])
    end = t_frame.max()

    rows = []
    for m in args.marks: # window [m - w/2, m + w/2], clamped into the run
        lo = min(max(m - args.window / 2, 0), end - args.window); hi = lo + args.window
        f = (t_frame >= lo) & (t_frame < hi) & (np.arange(len(pre)) > 0)
        d = (t_dec >= lo) & (t_dec < hi)
        rows.append(dict(mark=m, lo=lo, hi=hi, frames=int(f.sum()), replies=int(d.sum()), tokens=int(sum(e['units'] for e, k in zip(dec, d) if k)),
                         kv=int(np.median([e['kv_tokens'] for e, k in zip(pre, f) if k])),
                         tok_ms=1000 * s_dec[d].sum() / max(sum(e['units'] for e, k in zip(dec, d) if k), 1)))
        r = rows[-1]
        r['answer_len'] = r['tokens'] / max(r['replies'], 1)
        r['ms'] = [1000 * s_vis[f].mean(), 1000 * s_pre[f].mean(), r['tok_ms'] * r['answer_len']]

    plt.rcParams.update({'font.size': 20, 'figure.facecolor': SURF, 'axes.facecolor': SURF})
    names = ['vision encode', 'frame prefill', 'decode']
    ncol = 2; nrow = -(-len(rows) // ncol)
    fig = plt.figure(figsize=(24, 2.8 + 6.2 * nrow))
    head = 2.8 / (2.8 + 6.2 * nrow); cell_h = (1 - head) / nrow
    fig.text(0.01, 1 - 0.25 * head, f'Latency of an answering frame{" — " + args.label if args.label else ""}', color=INK, fontsize=26,
             fontweight='bold', va='top')
    fig.text(0.01, 1 - 0.55 * head, f'Vision encode + frame prefill of one frame + decode of its answer (ms per generated token × mean answer length),\n'
             f'averaged over a {args.window:g}-minute window of video around each mark.', color=MUTED, fontsize=19, va='top')
    for k, r in enumerate(rows): # each cell: pie on the left, its numbers on the right
        x0, y0 = (k % ncol) / ncol, 1 - head - (k // ncol + 1) * cell_h
        s = np.array(r['ms']); total = s.sum()
        a = fig.add_axes([x0 + 0.01, y0 + 0.04 * cell_h, 0.2, 0.8 * cell_h])
        wedges, _ = a.pie(s, colors=SERIES, startangle=90, counterclock=False, wedgeprops=dict(edgecolor=SURF, linewidth=3))
        big = int(np.argmax(s)) # only the largest slice is labelled on the pie; every share is listed beside it
        ang = np.deg2rad((wedges[big].theta1 + wedges[big].theta2) / 2)
        a.text(0.55 * np.cos(ang), 0.55 * np.sin(ang), f'{100 * s[big] / total:.2f}%', ha='center', va='center', color='white',
               fontsize=22, fontweight='bold')
        a.set_aspect('equal')
        t = fig.add_axes([x0 + 0.23, y0 + 0.04 * cell_h, 0.26, 0.8 * cell_h]); t.axis('off')
        t.text(0, 0.95, f'{r["mark"]:g} min  ({round(r["lo"] * 2) / 2:g}–{round(r["hi"] * 2) / 2:g} min)', color=INK, fontsize=22,
               fontweight='bold', va='top', transform=t.transAxes)
        for j, (n, v, c) in enumerate(zip(names, s, SERIES)):
            yy = 0.72 - 0.13 * j
            t.add_patch(plt.Rectangle((0, yy - 0.035), 0.045, 0.07, color=c, transform=t.transAxes, clip_on=False))
            t.text(0.07, yy, n, color=INK, va='center', transform=t.transAxes)
            t.text(0.72, yy, f'{v:.0f} ms', color=INK, va='center', ha='right', transform=t.transAxes)
            t.text(1.0, yy, f'{100 * v / total:.2f}%', color=INK, va='center', ha='right', transform=t.transAxes)
        t.text(0, 0.30, f'total {total:.0f} ms\ndecode = {r["answer_len"]:.1f} tokens × {r["tok_ms"]:.0f} ms\n'
               f'{r["replies"]} replies · ~{r["kv"] / 1000:.1f}k cached tokens', color=MUTED, fontsize=18, va='top', linespacing=1.4,
               transform=t.transAxes)
    out = args.out or args.growth_json.replace('_growth.json', '_breakdown.png')
    fig.savefig(out, dpi=160, bbox_inches='tight', pad_inches=0.2)
    print(out)
    for r in rows:
        s = np.array(r['ms'])
        print(f'{r["mark"]:g} min [{r["lo"]:g}-{r["hi"]:g}]: replies {r["replies"]}, answer {r["answer_len"]:.1f} tok x {r["tok_ms"]:.0f} ms, kv ~{r["kv"]}, '
              + ', '.join(f'{n} {v:.0f} ms ({100 * v / s.sum():.2f}%)' for n, v in zip(names, s)))

if __name__ == '__main__':
    main()
