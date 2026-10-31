# Log power to a CSV in the background while experiments run, one row per sample with a wall-clock (time.time()) timestamp:
# GPU board power from nvidia-smi, plus every power sensor the kernel exposes under /sys/class/hwmon (on the DGX Spark /
# Jetson these can include whole-module or rail power). profiling_scripts/energy_report.py integrates it per run.
# Usage: python -m profiling_scripts.power_log --probe                      # show which power sources exist, then exit
#        python -m profiling_scripts.power_log out.csv [--interval_ms 100]  # log until killed (SIGINT / SIGTERM)
import argparse, csv, glob, os, signal, subprocess, sys, time

def hwmon_sensors():
    """{column name: sysfs file} for every hwmon power reading (microwatts)."""
    sensors = {}
    for f in sorted(glob.glob('/sys/class/hwmon/hwmon*/power*_input') + glob.glob('/sys/class/hwmon/hwmon*/power*_average')):
        d = os.path.dirname(f)
        try:
            chip = open(os.path.join(d, 'name')).read().strip()
        except OSError:
            chip = os.path.basename(d)
        label_file = f.rsplit('_', 1)[0] + '_label'
        label = open(label_file).read().strip() if os.path.exists(label_file) else os.path.basename(f).rsplit('_', 1)[0]
        sensors[f'{chip}:{label}'.replace(' ', '_').replace(',', '')] = f
    return sensors

def read_watts(path):
    try:
        return int(open(path).read()) / 1e6
    except (OSError, ValueError):
        return float('nan')

def parse_smi(line):
    try:
        return float(line.strip().split(',')[0])
    except ValueError: # '[N/A]' or '[Not Supported]'
        return float('nan')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('out_csv', nargs='?')
    p.add_argument('--interval_ms', type=int, default=100)
    p.add_argument('--probe', action='store_true')
    args = p.parse_args()
    sensors = hwmon_sensors()
    smi = subprocess.Popen(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits', f'-lms', str(args.interval_ms)],
                           stdout=subprocess.PIPE, text=True, bufsize=1)
    if args.probe:
        for _ in range(5):
            line = smi.stdout.readline()
            print(f'nvidia-smi power.draw: {line.strip()!r} -> {parse_smi(line)} W')
        smi.terminate()
        print(f'{len(sensors)} hwmon power sensors' + (':' if sensors else ' (only the GPU reading will be logged)'))
        for k, f in sensors.items():
            print(f'  {k:40s} {read_watts(f):8.2f} W   ({f})')
        return
    if not args.out_csv:
        p.error('out_csv is required unless --probe')
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time', 'gpu_w'] + list(sensors))
        n = 0
        while not stop:
            line = smi.stdout.readline() # nvidia-smi paces the loop
            if not line:
                break
            w.writerow([f'{time.time():.3f}', parse_smi(line)] + [f'{read_watts(s):.3f}' for s in sensors.values()])
            n += 1
            if n % 50 == 0:
                f.flush()
    smi.terminate()
    print(f'power log: {n} samples -> {args.out_csv}', file=sys.stderr)

if __name__ == '__main__':
    main()
