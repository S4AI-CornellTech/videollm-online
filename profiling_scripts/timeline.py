from collections import defaultdict

PHASE_COLORS = {
    'vision_embed': '#4C72B0',
    'frame_prefill': '#DD8452',
    'decode': '#55A868',
    'busy_wait': '#C44E52',
}

def summarize_phases(phase_events):
    """
    phase_events: list of (phase_name, t_start, t_end, unit_count) using time.perf_counter() timestamps.
    Returns one row per phase_name with count/duration/unit aggregates.
    """
    groups = defaultdict(list)
    for name, t_start, t_end, unit_count in phase_events:
        groups[name].append((t_end - t_start, unit_count))

    rows = []
    for name, items in groups.items():
        durations = [d for d, _ in items]
        units = [u for _, u in items]
        count = len(items)
        total_duration = sum(durations)
        total_units = sum(units)
        rows.append({
            'phase': name,
            'count': count,
            'total_duration_s': total_duration,
            'mean_duration_s': total_duration / count,
            'total_units': total_units,
            'duration_per_unit_s': total_duration / total_units if total_units else float('nan'),
        })
    return sorted(rows, key=lambda r: -r['total_duration_s'])

def print_phase_summary(phase_events):
    rows = summarize_phases(phase_events)
    if not rows:
        print('No phase timing data recorded.')
        return
    header = f"{'phase':<15}{'count':>8}{'total_s':>12}{'mean_s':>12}{'units':>10}{'s/unit':>12}"
    print(header)
    print('-' * len(header))
    for r in rows:
        print(f"{r['phase']:<15}{r['count']:>8}{r['total_duration_s']:>12.4f}{r['mean_duration_s']:>12.6f}{r['total_units']:>10}{r['duration_per_unit_s']:>12.6f}")

    session_span = max(e[2] for e in phase_events) - min(e[1] for e in phase_events)
    accounted = sum(r['total_duration_s'] for r in rows)
    unaccounted = session_span - accounted
    print('-' * len(header))
    print(f"{'session_span':<15}{'':>8}{session_span:>12.4f}")
    print(f"{'unaccounted':<15}{'':>8}{unaccounted:>12.4f} ({unaccounted / session_span * 100:.1f}% of session — CPU-side work between phases: tokenizer calls, dict/list bookkeeping, etc.)")

def plot_phase_timeline(phase_events, out_path):
    """
    Single-row Gantt-style timeline: x-axis = seconds since the first recorded event's t_start,
    one colored horizontal bar per phase occurrence, in the order they actually happened.
    """
    if not phase_events:
        print('No phase timing data recorded; skipping timeline plot.')
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    t0 = min(e[1] for e in phase_events)
    t_max = max(e[2] for e in phase_events) - t0

    fig, ax = plt.subplots(figsize=(14, 2.5))
    for name, t_start, t_end, _ in phase_events:
        # fixed (not session-length-scaled) floor so a near-zero-duration phase (e.g. busy_wait)
        # still renders a sliver, without inflating real gaps between phases on long sessions
        duration = max(t_end - t_start, 0.002)
        ax.broken_barh([(t_start - t0, duration)], (0, 1), facecolors=PHASE_COLORS.get(name, '#888888'))

    ax.set_xlabel('Time since session start (s)')
    ax.set_xlim(0, t_max)
    ax.set_yticks([])
    present_phases = {e[0] for e in phase_events}
    handles = [mpatches.Patch(color=c, label=n) for n, c in PHASE_COLORS.items() if n in present_phases]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.45), ncol=len(handles))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Phase timeline plot saved to {out_path}')
