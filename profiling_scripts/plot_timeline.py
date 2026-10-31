# Phase timeline of a streaming session (vision encoding / frame prefill / decode over session time), from the
# <video>_phase_events.json that demo.cli writes with --phase_timing True. If the matching <video>_growth.json has
# reply details, degenerate replies are marked with red crosses above the strip and the sustained breakdown with a dashed line.
# Usage: python -m profiling_scripts.plot_timeline outputs/profiling/spark/coin_runA_phase_events.json --label "DGX Spark, run A"
import argparse, json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

INK, MUTED, GRID, SURF, RED = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb', '#e34948'
PHASES = {'vision_embed': ('#2a78d6', 'vision encoding'), 'frame_prefill': ('#eb6834', 'frame prefill'),
          'decode': ('#1baf7a', 'decode (generating replies)')}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase_events_json')
    p.add_argument('--label', default='')
    p.add_argument('--out', default=None)
    args = p.parse_args()
    ph = json.load(open(args.phase_events_json))
    t0 = min(e[1] for e in ph)
    span = (max(e[2] for e in ph) - t0) / 60
    growth_path = args.phase_events_json.replace('_phase_events.json', '_growth.json')
    rows = [e for e in json.load(open(growth_path)) if e['phase'] == 'decode'] if os.path.exists(growth_path) else []
    video_min = max((e['video_time'] for e in rows), default=0) / 60

    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED,
                         'text.color': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF})
    fig, ax = plt.subplots(figsize=(13, 2.9))
    for name, a, b, _ in ph:
        if name in PHASES:
            ax.broken_barh([((a - t0) / 60, max(b - a, 0.002) / 60)], (0, 1), facecolors=PHASES[name][0])
    ax.set_xlim(0, span); ax.set_ylim(0, 1.32); ax.set_yticks([])
    ax.set_xlabel('Session time (minutes)')
    title = f'VideoLLM-online 8B{" on " + args.label if args.label else ""}: phase timeline'
    if video_min:
        title += f' of a {video_min:.0f}-minute stream ({span:.1f}-minute session)'
    ax.set_title(title, loc='left', fontsize=12, fontweight='bold')
    handles = [mpatches.Patch(color=c, label=l) for c, l in PHASES.values()]

    # decode phase events and decode growth rows are logged in the same order: the i-th decode event is the i-th reply
    decs = [e for e in ph if e[0] == 'decode']
    if rows and 'ended_with_eos' in rows[0] and len(rows) == len(decs):
        from profiling_scripts.degeneration_report import is_degenerate, sustained_onset
        when = [(d[1] - t0) / 60 for d in decs]
        bad = [w for w, e in zip(when, rows) if is_degenerate(e)]
        ok = [w for w, e in zip(when, rows) if not is_degenerate(e)]
        ax.plot(ok, [1.16] * len(ok), '|', color=MUTED, ms=9, mew=1.2)
        ax.plot(bad, [1.16] * len(bad), 'x', color=RED, ms=7, mew=2)
        handles += [Line2D([], [], marker='|', ls='', color=MUTED, ms=9, mew=1.2, label=f'normal reply ({len(ok)})'),
                    Line2D([], [], marker='x', ls='', color=RED, ms=7, mew=2, label=f'degenerate reply ({len(bad)})')]
        onset = sustained_onset(rows)
        if onset:
            t = when[rows.index(onset)]
            ax.axvline(t, color=RED, lw=1.5, ls='--')
            ax.text(t, 0.5, f'  replies break down from here on: video minute {onset["video_time"] / 60:.0f}, '
                    f'{onset["kv_tokens"]:,} cached tokens  ', ha='right', va='center', fontsize=9, color=INK,
                    bbox=dict(boxstyle='round,pad=0.3', fc=SURF, ec=RED, lw=1))
        else:
            ax.text(span * 0.99, 0.5, 'no breakdown of replies', ha='right', va='center', fontsize=9, color=INK,
                    bbox=dict(boxstyle='round,pad=0.3', fc=SURF, ec=GRID, lw=1))
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.32), ncol=len(handles), frameon=False)
    out = args.out or args.phase_events_json.replace('_phase_events.json', '_timeline.png')
    fig.savefig(out, dpi=160, bbox_inches='tight')
    print(f'saved {out}')

if __name__ == '__main__':
    main()
