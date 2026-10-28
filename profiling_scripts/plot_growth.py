# Plot how streaming cost grows with video length, from the <video>_growth.json that demo.cli writes with --phase_timing True.
# Three stacked panels on the same minute axis: KV cache size, frame prefill time, and decode time per token,
# each with a linear fit (dashed) extrapolated to --until minutes, and the real-time limit for prefill.
# Usage: python -m profiling_scripts.plot_growth outputs/profiling/coin_long_60min_growth.json --label "DGX Spark" [--until 60] [--fps 2]
import argparse, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, INK, MUTED, GRID, SURF, REF = '#2a78d6', '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb', '#8a8984'
KV_BYTES_PER_TOKEN = 131072 # Llama-3-8B: 2 (K, V) x 32 layers x 8 KV heads x 128 dims x 2 bytes (bf16)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('growth_json')
    p.add_argument('--label', default='')
    p.add_argument('--until', type=float, default=60, help='extrapolate the fits to this many minutes')
    p.add_argument('--fps', type=float, default=2)
    p.add_argument('--vision_ms', type=float, default=None, help='per-frame vision encoding time; default: read from <video>_phase_events.json')
    p.add_argument('--out', default=None)
    args = p.parse_args()
    events = json.load(open(args.growth_json))
    pre = [e for e in events if e['phase'] == 'frame_prefill']
    dec = [e for e in events if e['phase'] == 'decode' and e['units'] > 0]
    vision_ms = args.vision_ms
    if vision_ms is None:
        try:
            ph = json.load(open(args.growth_json.replace('_growth.json', '_phase_events.json')))
            vis = [(t1 - t0) for name, t0, t1, n in ph if name == 'vision_embed']
            vision_ms = 1000 * sum(vis) / len(vis)
        except (OSError, ZeroDivisionError):
            vision_ms = 0

    # per-minute means for prefill (2 fps = 120 frames a minute, so single frames are noisy); every decode is its own point
    pm = {}
    for e in pre:
        pm.setdefault(int(e['video_time'] // 60), []).append(e)
    minutes = np.array(sorted(pm))
    kv = np.array([max(e['kv_tokens'] for e in pm[m]) for m in minutes])
    pre_ms = np.array([1000 * np.mean([e['seconds'] for e in pm[m]]) for m in minutes])
    full = np.array([len(pm[m]) >= 0.9 * 60 * args.fps for m in minutes]) # fit only on complete minutes
    dec_min = np.array([e['video_time'] / 60 for e in dec]); dec_ms = np.array([1000 * e['seconds'] / e['units'] for e in dec])
    dec_kv = np.array([e['kv_tokens'] for e in dec])

    kv_a, kv_b = np.polyfit(minutes[full], kv[full], 1)
    pr_a, pr_b = np.polyfit(kv[full], pre_ms[full], 1)
    # fit decode on typical replies: skip one-time warm-up and very short replies, whose fixed costs dominate the per-token time
    typical = np.array([e['units'] >= 3 for e in dec], dtype=bool) & (dec_min > 0.5)
    if typical.any():
        typical &= dec_ms <= 2 * np.median(dec_ms[typical])
    dc_a, dc_b = np.polyfit(dec_kv[typical], dec_ms[typical], 1) if typical.sum() >= 3 else (np.nan, np.nan)
    mx = np.linspace(0, args.until, 241); kvx = kv_a * mx + kv_b
    budget = 1000 / args.fps - vision_ms
    cross = ((budget - pr_b) / pr_a - kv_b) / kv_a if pr_a > 0 else np.inf
    measured_until = minutes[-1] + 1

    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED,
                         'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': SURF, 'axes.facecolor': SURF})
    fig, axes = plt.subplots(3, 1, figsize=(9, 9.6), sharex=True, gridspec_kw={'hspace': 0.32})
    fig.suptitle(f'VideoLLM-online{" on " + args.label if args.label else ""}: cost of streaming a long video at {args.fps:g} fps',
                 x=0.07, ha='left', fontsize=12.5, fontweight='bold')
    fig.text(0.07, 0.935, f'Measured over {measured_until:.0f} min (solid / dots), linear fit extrapolated to {args.until:g} min (dashed).', fontsize=9, color=MUTED)
    for ax in axes:
        ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
        if args.until > measured_until:
            ax.axvspan(measured_until, args.until + 1, color='#f2f1ee', zorder=0)

    a1, a2, a3 = axes
    a1.plot(mx, kvx / 1000, color=BLUE, lw=2, ls=(0, (5, 3)))
    a1.plot(minutes, kv / 1000, color=BLUE, lw=2)
    a1.axhline(8.192, color=REF, lw=1, ls=':')
    a1.text(args.until, 8.192 * 1.25, "Llama-3's trained context (8,192 tokens)", ha='right', fontsize=8.5, color=MUTED)
    a1.set_ylabel('KV cache (k tokens)')
    a1.set_title(f'KV cache: ~{kv_a:,.0f} tokens ({kv_a * KV_BYTES_PER_TOKEN / 1e9:.2f} GB) per minute; '
                 f'{(kv_a * args.until + kv_b) * KV_BYTES_PER_TOKEN / 1e9:.1f} GB at {args.until:g} min', loc='left', fontsize=10.5)

    a2.plot(mx, pr_a * kvx + pr_b, color=BLUE, lw=2, ls=(0, (5, 3)))
    a2.plot(minutes, pre_ms, color=BLUE, lw=2)
    a2.axhline(budget, color=REF, lw=1, ls=':')
    a2.text(0.5, budget * 1.04, f'real-time limit: {budget:.0f} ms/frame ({1000 / args.fps:.0f} ms minus {vision_ms:.0f} ms vision)', fontsize=8.5, color=MUTED)
    if cross <= args.until:
        a2.plot([cross], [budget], 'o', color=BLUE, ms=6, mec=SURF, mew=2)
        a2.annotate(f'falls behind live video at ~minute {cross:.0f}', (cross, budget), xytext=(10, -22), textcoords='offset points', fontsize=8.5)
    a2.set_ylabel('Prefill (ms / frame)'); a2.set_ylim(0, max(budget * 1.25, (pr_a * kvx[-1] + pr_b) * 1.1, pre_ms.max() * 1.1))
    a2.set_title(f'Frame prefill: {pr_b:.0f} ms + {pr_a * 1000:.1f} ms per 1k cached tokens', loc='left', fontsize=10.5)

    if not np.isnan(dc_a):
        a3.plot(mx, dc_a * kvx + dc_b, color=BLUE, lw=2, ls=(0, (5, 3)))
        a3.set_title(f'Decode: {dc_b:.0f} ms + {dc_a * 1000:.1f} ms per 1k cached tokens, per generated token', loc='left', fontsize=10.5)
    a3.plot(dec_min[typical], dec_ms[typical], 'o', color=BLUE, ms=5, mec=SURF, mew=1.5)
    a3.plot(dec_min[~typical], dec_ms[~typical], 'o', color=BLUE, alpha=0.3, ms=5, mec=SURF, mew=1.5) # excluded from the fit
    top = max(dec_ms[typical].max() if typical.any() else dec_ms.max(), (dc_a * kvx[-1] + dc_b) if not np.isnan(dc_a) else 0) * 1.3
    a3.set_ylim(0, top)
    if (dec_ms > top).any():
        a3.text(0.5, top * 0.97, f'{(dec_ms > top).sum()} warm-up/short replies above {top:.0f} ms not shown', fontsize=8, color=MUTED, va='top')
    a3.set_ylabel('Decode (ms / token)')
    a3.set_xlabel('Video time (minutes)'); a3.set_xlim(0, args.until + 1)
    out = args.out or args.growth_json.replace('_growth.json', '_growth.png')
    fig.savefig(out, dpi=160, bbox_inches='tight')
    print(f'KV: {kv_a:.0f} tokens/min | prefill: {pr_b:.0f} ms + {pr_a * 1000:.2f} ms/1k tokens | decode: {dc_b:.0f} ms + {dc_a * 1000:.2f} ms/1k tokens per token')
    print(f'real-time limit {budget:.0f} ms/frame -> falls behind at ~minute {cross:.1f}' if np.isfinite(cross) else 'prefill does not grow')
    print(f'saved {out}')

if __name__ == '__main__':
    main()
