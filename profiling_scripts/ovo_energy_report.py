# Time, energy and accuracy per frame rate for scripts/spark/ovo_qwen3vl.sh: integrates power.csv (profiling_scripts/power_log.py)
# over each question's wall-clock window (t_start / t_end that qwen3vl_fps_eval.py logs), split into video loading
# (CPU decode + resize, the first load_s seconds) and the model (vision encode + prefill + decode), and compares with
# the idle baseline in idle.json ('above idle' = what the question added on top of the machine idling).
# Optionally scores the same questions from a reference run (e.g. the B300 results) to check that accuracy carries over.
# Usage: python -m profiling_scripts.ovo_energy_report outputs/profiling/spark_ovo [--column gpu_w] [--ref_dir outputs/ovo]
import argparse, glob, json, os
import numpy as np
from profiling_scripts.energy_report import load_power, energy

def correct(r):
    return int(r.get('response') is not None and r['ground_truth'] in r['response'])

def main():
    p = argparse.ArgumentParser()
    p.add_argument('out_dir')
    p.add_argument('--column', default='gpu_w', help='power column of power.csv to integrate')
    p.add_argument('--ref_dir', default=None, help='folder with qwen3vl8b_<fps>fps.jsonl from another machine')
    a = p.parse_args()
    header, rows = load_power(os.path.join(a.out_dir, 'power.csv'))
    t, w = rows[:, 0], rows[:, header.index(a.column)]
    idle = json.load(open(os.path.join(a.out_dir, 'idle.json')))
    idle_w = energy(t, w, idle['idle_start'], idle['idle_end'])[1]
    print(f'power column {a.column}; idle {idle_w:.1f} W\n')
    cols = ['fps', 'n', 'acc %', 'ref acc %', 'tokens', 'load s', 'model s', 'peak GB', 'J/question', 'model J', 'above idle J', 'mean W']
    print(''.join(f'{c:>13}' for c in cols))
    for path in sorted(glob.glob(os.path.join(a.out_dir, 'qwen3vl8b_*fps.jsonl')), key=lambda f: float(f.rsplit('_', 1)[1][:-len('fps.jsonl')])):
        rs = [r for r in map(json.loads, open(path)) if r.get('response') is not None and 't_end' in r]
        if not rs:
            continue
        e_all, e_model, above, watts = [], [], [], []
        for r in rs:
            j, mw = energy(t, w, r['t_start'], r['t_end'])
            jm, _ = energy(t, w, r['t_start'] + r['load_s'], r['t_end'])
            e_all.append(j); e_model.append(jm); watts.append(mw)
            above.append(j - idle_w * (r['t_end'] - r['t_start']))
        ref = ''
        if a.ref_dir:
            rp = os.path.join(a.ref_dir, os.path.basename(path))
            if os.path.exists(rp):
                byid = {r['id']: r for r in map(json.loads, open(rp))}
                ref = f'{100 * np.mean([correct(byid[r["id"]]) for r in rs if r["id"] in byid]):.1f}'
        vals = [f'{rs[0]["fps"]:g}', len(rs), f'{100 * np.mean([correct(r) for r in rs]):.1f}', ref or '-',
                f'{np.mean([r["input_tokens"] for r in rs]):.0f}', f'{np.mean([r["load_s"] for r in rs]):.2f}',
                f'{np.mean([r["generate_s"] for r in rs]):.2f}', f'{max(r["peak_gb"] for r in rs):.1f}',
                f'{np.nanmean(e_all):.0f}', f'{np.nanmean(e_model):.0f}', f'{np.nanmean(above):.0f}', f'{np.nanmean(watts):.1f}']
        print(''.join(f'{v:>13}' for v in vals))

if __name__ == '__main__':
    main()
