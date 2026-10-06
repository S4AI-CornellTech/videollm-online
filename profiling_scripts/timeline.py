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

def summarize_phase_energy(phase_events, power_meter):
    """
    For each recorded phase occurrence, integrate power over its [t_start, t_end]
    window (via power_meter.measurement_between) and aggregate by phase_name.
    """
    groups = defaultdict(list)
    for name, t_start, t_end, unit_count in phase_events:
        energy_j, mean_w, n_real = power_meter.measurement_between(t_start, t_end)
        groups[name].append((energy_j, unit_count, n_real))

    rows = []
    for name, items in groups.items():
        total_energy = sum(e for e, _, _ in items)
        total_units = sum(u for _, u, _ in items)
        total_real_samples = sum(n for _, _, n in items)
        rows.append({
            'phase': name,
            'count': len(items),
            'total_energy_j': total_energy,
            'energy_per_unit_j': total_energy / total_units if total_units else float('nan'),
            'n_real_samples': total_real_samples,
        })
    return sorted(rows, key=lambda r: -r['total_energy_j'])

def print_phase_energy_summary(phase_events, power_meter):
    rows = summarize_phase_energy(phase_events, power_meter)
    if not rows:
        print('No phase energy data recorded.')
        return
    total_energy = sum(r['total_energy_j'] for r in rows)
    header = f"{'phase':<15}{'count':>8}{'energy_J':>12}{'share':>8}{'J/unit':>12}{'real_samp':>10}"
    print(header)
    print('-' * len(header))
    for r in rows:
        share = r['total_energy_j'] / total_energy * 100 if total_energy else 0
        print(f"{r['phase']:<15}{r['count']:>8}{r['total_energy_j']:>12.3f}{share:>7.1f}%{r['energy_per_unit_j']:>12.5f}{r['n_real_samples']:>10}")
    n_thin = sum(r['n_real_samples'] < r['count'] for r in rows)
    if n_thin:
        print(f"(Note: GB10's power sensor updates at roughly ~1Hz — most individual phase windows are")
        print(f" shorter than that, so most phase-occurrence energies are boundary-interpolated, not")
        print(f" independently measured. Trust the aggregate/share numbers over any single occurrence.)")

def plot_phase_timeline_with_power(phase_events, power_samples, out_path, title=None):
    """
    Two stacked panels sharing an x-axis (seconds since session start):
    top = raw power draw trace (W), bottom = the same phase-color Gantt strip
    as plot_phase_timeline.
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

    fig, (ax_power, ax_phase) = plt.subplots(
        2, 1, figsize=(14, 4.5), sharex=True,
        gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.08},
    )
    suptitle = fig.suptitle(title) if title else None

    if power_samples:
        xs = [s.perf_t - t0 for s in power_samples]
        ys = [s.watts for s in power_samples]
        # steps-post: GB10's power sensor holds each reading for ~1s regardless of
        # our polling rate, so a step plot reflects the real sensor behavior more
        # honestly than a smoothly-interpolated line would.
        ax_power.plot(xs, ys, color='#333333', linewidth=1.2, drawstyle='steps-post')
        ax_power.set_ylabel('Power (W)')
        ax_power.grid(True, alpha=0.3)
    else:
        ax_power.text(0.5, 0.5, 'No power samples', ha='center', va='center', transform=ax_power.transAxes)

    for name, t_start, t_end, _ in phase_events:
        duration = max(t_end - t_start, 0.002)
        ax_phase.broken_barh([(t_start - t0, duration)], (0, 1), facecolors=PHASE_COLORS.get(name, '#888888'))
    ax_phase.set_yticks([])
    ax_phase.set_xlabel('Time since session start (s)')
    ax_phase.set_xlim(0, t_max)

    present_phases = {e[0] for e in phase_events}
    handles = [mpatches.Patch(color=c, label=n) for n, c in PHASE_COLORS.items() if n in present_phases]
    legend = ax_phase.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.55), ncol=len(handles))

    fig.tight_layout()
    extra_artists = [legend] + ([suptitle] if suptitle else [])
    fig.savefig(out_path, dpi=150, bbox_inches='tight', bbox_extra_artists=extra_artists)
    plt.close(fig)
    print(f'Phase+power timeline plot saved to {out_path}')

def plot_phase_timeline_overlay(phase_events, power_samples, out_path, title=None):
    """
    Single axes, twin y-axis: the power (W) line drawn directly on top of/through
    the colored phase-block strip, both sharing the same x-axis (time since session start).
    """
    if not phase_events:
        print('No phase timing data recorded; skipping timeline plot.')
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    t0 = min(e[1] for e in phase_events)
    t_max = max(e[2] for e in phase_events) - t0

    fig, ax_phase = plt.subplots(figsize=(14, 3.5))
    if title:
        ax_phase.set_title(title)

    for name, t_start, t_end, _ in phase_events:
        duration = max(t_end - t_start, 0.002)
        ax_phase.broken_barh([(t_start - t0, duration)], (0, 1), facecolors=PHASE_COLORS.get(name, '#888888'))
    ax_phase.set_yticks([])
    ax_phase.set_xlabel('Time since session start (s)')
    ax_phase.set_xlim(0, t_max)
    ax_phase.set_ylim(0, 1)

    ax_power = ax_phase.twinx()
    # draw the power line above the phase blocks, and make the twin axes' own
    # background transparent so it doesn't paint over the colored blocks
    ax_power.set_zorder(ax_phase.get_zorder() + 1)
    ax_power.patch.set_visible(False)
    if power_samples:
        xs = [s.perf_t - t0 for s in power_samples]
        ys = [s.watts for s in power_samples]
        ax_power.plot(xs, ys, color='black', linewidth=1.5, drawstyle='steps-post')
        ax_power.set_ylabel('Power (W)')
    else:
        ax_power.text(0.5, 0.5, 'No power samples', ha='center', va='center', transform=ax_power.transAxes)

    present_phases = {e[0] for e in phase_events}
    phase_handles = [mpatches.Patch(color=c, label=n) for n, c in PHASE_COLORS.items() if n in present_phases]
    power_handle = mlines.Line2D([], [], color='black', linewidth=1.5, label='power (W)')
    legend = ax_phase.legend(handles=phase_handles + [power_handle], loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=len(phase_handles) + 1)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', bbox_extra_artists=[legend])
    plt.close(fig)
    print(f'Overlay (twin-axis) timeline saved to {out_path}')

def plot_phase_timeline(phase_events, out_path, title=None):
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
    if title:
        ax.set_title(title)
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
    legend = ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.45), ncol=len(handles))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', bbox_extra_artists=[legend])
    plt.close(fig)
    print(f'Phase timeline plot saved to {out_path}')
