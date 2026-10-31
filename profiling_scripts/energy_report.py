# Energy per run and per FPS from scripts/spark/energy_fps.sh output: integrates power.csv (profiling_scripts/power_log.py)
# over each run's streaming window (the 'session' wall-clock times demo.cli saves in <run>.json), splits it over the
# vision / prefill / decode phases from <run>_phase_events.json, and compares with the idle baseline in idle.json.
# demo.cli streams as fast as it can, so a run takes less time than its video; 'real-time J/min' adds idle power for the
# rest of the video's duration, i.e. the energy if frames arrived live and the device idled in between.
# Each run is measured from the end of its first frame, which can carry one-time start-up costs.
# Usage: python -m profiling_scripts.energy_report outputs/profiling/energy [--column gpu_w]
import argparse, csv, glob, json, os
from collections import defaultdict
import numpy as np
trapezoid = getattr(np, 'trapezoid', None) or np.trapz # numpy 2 renamed it

PHASES = ('vision_embed', 'frame_prefill', 'decode')

def load_power(path):
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        rows = np.array([[float(x) for x in row] for row in r if len(row) == len(header)])
    return header, rows

def energy(t, w, t0, t1):
    """Joules between t0 and t1 (trapezoid over the samples inside, NaN samples dropped) and the mean watts."""
    m = (t >= t0) & (t <= t1) & ~np.isnan(w)
    if m.sum() < 2:
        return float('nan'), float('nan')
    return float(trapezoid(w[m], t[m])), float(w[m].mean())

def main():
    p = argparse.ArgumentParser()
    p.add_argument('out_dir')
    p.add_argument('--column', default='gpu_w', help='power column to report (see power_sources.txt / the CSV header)')
    p.add_argument('--plot', default=None)
    args = p.parse_args()
    header, P = load_power(os.path.join(args.out_dir, 'power.csv'))
    t = P[:, 0]
    cols = {h: P[:, i] for i, h in enumerate(header) if i}
    w = cols[args.column]
    idle = json.load(open(os.path.join(args.out_dir, 'idle.json')))
    idle_w = {c: energy(t, v, idle['idle_start'], idle['idle_end'])[1] for c, v in cols.items()}
    print(f'idle power ({args.column}): {idle_w[args.column]:.1f} W' +
          ''.join(f' | {c}: {v:.1f} W' for c, v in idle_w.items() if c != args.column))

    runs = []
    for path in sorted(glob.glob(os.path.join(args.out_dir, '*fps.json'))):
        h = json.load(open(path))
        s = h.get('session')
        if not s:
            continue
        name = os.path.basename(path)[:-5]
        ev_path = path[:-5] + '_phase_events.json'
        ev = json.load(open(ev_path)) if os.path.exists(ev_path) else []
        # measure from the end of the first frame: it can still carry a one-time cost (e.g. the GPU waking from idle)
        prefills = [e_ for e_ in ev if e_[0] == 'frame_prefill']
        start, video_s = s['wall_start'], s['video_seconds']
        if len(prefills) > 1:
            start = s['wall_start'] + (prefills[0][2] - s['perf_start'])
            video_s -= 1 / float(h['frame_fps'])
        e, mean_w = energy(t, w, start, s['wall_end'])
        compute_s = s['wall_end'] - start
        # energy per phase: each power sample is charged to the phase running at that moment
        phase_j = defaultdict(float)
        if ev:
            m = (t >= start) & (t <= s['wall_end']) & ~np.isnan(w)
            ts, ws = t[m], w[m]
            dt = np.gradient(ts) if len(ts) > 1 else np.zeros_like(ts)
            starts = np.array([s['wall_start'] + (e_[1] - s['perf_start']) for e_ in ev])
            ends = np.array([s['wall_start'] + (e_[2] - s['perf_start']) for e_ in ev])
            k = np.searchsorted(starts, ts, side='right') - 1
            for ti, wi, di, ki in zip(ts, ws, dt, k):
                phase = ev[ki][0] if ki >= 0 and ti <= ends[ki] else 'other'
                phase_j[phase if phase in PHASES else 'other'] += wi * di
        replies = sum(1 for c in h['conversation'] if c.get('role') == 'assistant')
        runs.append(dict(name=name, fps=float(h['frame_fps']), video_s=video_s, compute_s=compute_s, mean_w=mean_w,
                         energy_j=e, phase_j=dict(phase_j), replies=replies))
    if not runs:
        print('no finished runs'); return

    print('\n%-26s %6s %8s %9s %7s %9s %9s %8s' % ('run', 'fps', 'video_s', 'compute_s', 'mean_W', 'energy_J', 'J/vid_min', 'replies'))
    for r in runs:
        print('%-26s %6g %8.0f %9.1f %7.1f %9.0f %9.1f %8d' % (r['name'], r['fps'], r['video_s'], r['compute_s'], r['mean_w'],
              r['energy_j'], r['energy_j'] / (r['video_s'] / 60), r['replies']))

    by = defaultdict(list)
    for r in runs:
        by[r['fps']].append(r)
    iw = idle_w[args.column]
    print(f'\nper fps (sum over clips; {args.column}, idle {iw:.1f} W)')
    print('%6s %6s %9s %10s %7s %10s %13s %15s   %s' % ('fps', 'clips', 'video_min', 'compute_s', 'mean_W', 'J/vid_min',
          'above_idle', 'real-time J/min', 'energy share vision/prefill/decode/other'))
    summary = []
    for fps in sorted(by):
        rs = by[fps]
        vid_min = sum(r['video_s'] for r in rs) / 60
        comp = sum(r['compute_s'] for r in rs)
        ej = sum(r['energy_j'] for r in rs)
        rt = ej + iw * max(sum(r['video_s'] for r in rs) - comp, 0) # idle for the rest of the video's real duration
        ph = defaultdict(float)
        for r in rs:
            for k, v in r['phase_j'].items():
                ph[k] += v
        tot = sum(ph.values()) or float('nan')
        share = '/'.join(f'{100 * ph[k] / tot:.0f}' for k in PHASES + ('other',))
        print('%6g %6d %9.1f %10.0f %7.1f %10.1f %13.1f %15.1f   %s %%' % (fps, len(rs), vid_min, comp, ej / comp, ej / vid_min,
              (ej - iw * comp) / vid_min, rt / vid_min, share))
        summary.append((fps, ej / vid_min, rt / vid_min, ej / comp, comp / (vid_min * 60)))

    out = args.plot or os.path.join(args.out_dir, 'energy_vs_fps.png')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed: skipped the plot'); return
    fps, jmin, rtmin, mw, duty = map(np.array, zip(*summary))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.plot(fps, jmin, 'o-', color='#2a78d6', label='processing only (as fast as possible)')
    a1.plot(fps, rtmin, 's--', color='#eb6834', label='real-time stream (idle between frames)')
    a1.set_xscale('log', base=2); a1.set_xticks(fps, [f'{f:g}' for f in fps])
    a1.set_xlabel('frames per second'); a1.set_ylabel('energy per minute of video (J)'); a1.legend(frameon=False, fontsize=8)
    a1.set_ylim(bottom=0); a1.set_title(f'Energy ({args.column})', loc='left')
    a2.plot(fps, mw, 'o-', color='#1baf7a')
    a2.axhline(iw, color='#8a8984', ls=':', lw=1); a2.text(fps[0], iw * 1.03, f'idle {iw:.0f} W', fontsize=8, color='#52514e')
    for f_, m_, d_ in zip(fps, mw, duty):
        a2.annotate(f'busy {100 * d_:.0f}% of real time', (f_, m_), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=8)
    a2.set_xscale('log', base=2); a2.set_xticks(fps, [f'{f:g}' for f in fps])
    a2.set_xlabel('frames per second'); a2.set_ylabel('mean power while processing (W)'); a2.set_ylim(bottom=0)
    a2.set_title('Power', loc='left')
    fig.suptitle('VideoLLM-online 8B on DGX Spark: energy vs frame rate', x=0.02, ha='left', fontweight='bold')
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches='tight')
    print(f'saved {out}')

if __name__ == '__main__':
    main()
