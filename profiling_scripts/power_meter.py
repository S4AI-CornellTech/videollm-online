import subprocess, threading, time, collections

PowerSample = collections.namedtuple('PowerSample', ['perf_t', 'wall_t', 'watts'])

class NvidiaSmiPowerSampler:
    """Background-thread GPU power poller: a fresh single-shot `nvidia-smi` call per
    sample. nvidia-smi's own `--loop-ms` continuous-query mode was found (on this
    machine's driver, 580.95.05) to get stuck repeating its first reading for the
    rest of the subprocess's lifetime instead of re-querying the sensor -- confirmed
    reproducibly across two full runs -- so we deliberately avoid it."""

    def __init__(self, device_index=0, interval_ms=50):
        self.device_index = device_index
        self.interval_s = interval_ms / 1000.0
        self.samples = []
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _poll_loop(self):
        cmd = ['nvidia-smi', f'--id={self.device_index}', '--query-gpu=power.draw', '--format=csv,noheader,nounits']
        while not self._stop_event.is_set():
            t_perf = time.perf_counter()
            t_wall = time.time()
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=1).stdout.strip()
                watts = float(out)
                self.samples.append(PowerSample(t_perf, t_wall, watts))
            except (ValueError, subprocess.TimeoutExpired):
                pass  # e.g. 'N/A' or a transient hiccup; skip this sample
            self._stop_event.wait(self.interval_s)

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)


def integrate_energy(samples, start_t, end_t):
    """
    Trapezoidal integration of power (W) over [start_t, end_t] (perf_counter seconds).
    Boundary-point synthesis: if the window contains 0-1 real samples, a synthetic
    sample is inserted at start_t/end_t using the nearest real sample's wattage, so
    short windows (a single frame-prefill call, a single decode episode) don't
    spuriously integrate to zero just because the ~20Hz sampler happened to land
    outside the window.
    Returns (energy_joules, mean_watts, n_real_samples_in_window).
    """
    if not samples:
        return 0.0, 0.0, 0

    in_window = [s for s in samples if start_t <= s.perf_t <= end_t]
    n_real = len(in_window)

    if not in_window:
        nearest = min(samples, key=lambda s: min(abs(s.perf_t - start_t), abs(s.perf_t - end_t)))
        points = [(start_t, nearest.watts), (end_t, nearest.watts)]
    else:
        before = [s for s in samples if s.perf_t < start_t]
        after = [s for s in samples if s.perf_t > end_t]
        start_w = before[-1].watts if before else in_window[0].watts
        end_w = after[0].watts if after else in_window[-1].watts
        points = [(start_t, start_w)] + [(s.perf_t, s.watts) for s in in_window] + [(end_t, end_w)]

    energy = 0.0
    for (t0, w0), (t1, w1) in zip(points[:-1], points[1:]):
        energy += (w0 + w1) / 2.0 * max(0.0, t1 - t0)
    span = end_t - start_t
    mean_w = energy / span if span > 0 else points[0][1]
    return energy, mean_w, n_real


class PowerMeter:
    """Context-manager wrapper: one running sampler for a whole session, sliced by
    timestamp after the fact via measurement_between()."""

    def __init__(self, device_index=0, interval_ms=50):
        self.sampler = NvidiaSmiPowerSampler(device_index=device_index, interval_ms=interval_ms)

    def __enter__(self):
        self.sampler.start()
        return self

    def __exit__(self, *exc):
        self.sampler.stop()

    @property
    def samples(self):
        return self.sampler.samples

    def measurement_between(self, start_t, end_t):
        return integrate_energy(self.samples, start_t, end_t)
