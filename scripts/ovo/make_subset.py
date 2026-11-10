# Pick a fixed subset of OVO-Bench backward + real-time questions for small-device runs (DGX Spark): N per task, spread
# evenly over clip length, keeping only clips whose prompt at --max_fps stays under --max_tokens (estimated from a run
# that logged input_tokens at a known fps, e.g. the B300 2 fps run). Writes one id per line.
# Usage: python scripts/ovo/make_subset.py outputs/ovo/qwen3vl8b_2fps.jsonl --out scripts/ovo/spark_subset.txt [--per_task 12]
import argparse, json
from collections import defaultdict

def main():
    p = argparse.ArgumentParser()
    p.add_argument('ref', help='results JSONL with input_tokens')
    p.add_argument('--out', required=True)
    p.add_argument('--per_task', type=int, default=12)
    p.add_argument('--max_fps', type=float, default=2)
    p.add_argument('--max_tokens', type=int, default=120000)
    a = p.parse_args()
    by_task = defaultdict(list)
    for r in map(json.loads, open(a.ref)):
        if r.get('input_tokens') and r['input_tokens'] * a.max_fps / r['fps'] <= a.max_tokens:
            by_task[r['task']].append(r)
    ids = []
    for t, rows in sorted(by_task.items()):
        rows.sort(key=lambda r: (r['realtime'], r['id']))
        n = min(a.per_task, len(rows))
        pick = [rows[round(k * (len(rows) - 1) / max(n - 1, 1))] for k in range(n)] # evenly spaced over clip length
        ids += [r['id'] for r in pick]
        print(f'{t}: {n} of {len(rows)}, clips {pick[0]["realtime"]:.0f}-{pick[-1]["realtime"]:.0f} s')
    open(a.out, 'w').write('\n'.join(map(str, sorted(set(ids)))) + '\n')
    print(f'{len(set(ids))} ids -> {a.out}')

if __name__ == '__main__':
    main()
