# Score qwen3vl_fps_eval.py results the way OVO-Bench's score.py does (correct if the ground-truth letter appears in the
# response; category average = mean of its task accuracies), one column per results file, plus accuracy by clip length
# and the cost columns (frames, visual tokens, generate time, peak memory).
# Usage: python scripts/ovo/score_fps.py outputs/ovo/qwen3vl8b_0.5fps.jsonl outputs/ovo/qwen3vl8b_1fps.jsonl ... [--common]
#   --common: drop from every file the questions any file skipped (skipped=over_context), so columns compare the same set
import json, sys
from collections import defaultdict

BACKWARD = ['EPM', 'ASI', 'HLD']
REALTIME = ['OCR', 'ACR', 'ATR', 'STU', 'FPD', 'OJR']
LENGTH_BINS = [(0, 60), (60, 180), (180, 600), (600, 10**9)]

def correct(r):
    return int(r.get('response') is not None and r['ground_truth'] in r['response'])

def mean(x):
    return sum(x) / len(x) if x else float('nan')

def main(paths, common=False):
    data = {p: [json.loads(l) for l in open(p) if l.strip()] for p in paths}
    skipped = {r['id'] for rows in data.values() for r in rows if r.get('skipped')}
    if common and skipped:
        print(f'--common: {len(skipped)} questions skipped by some run are dropped from all columns')
    cols = []
    for path in paths:
        rows = [r for r in data[path] if not (common and r['id'] in skipped)]
        by_task = defaultdict(list)
        for r in rows:
            by_task[r['task']].append(correct(r))
        acc = {t: 100 * mean(v) for t, v in by_task.items()}
        back = mean([acc[t] for t in BACKWARD if t in acc])
        real = mean([acc[t] for t in REALTIME if t in acc])
        by_len = {b: 100 * mean([correct(r) for r in rows if b[0] <= r['realtime'] < b[1]]) for b in LENGTH_BINS}
        ok = [r for r in rows if r.get('response') is not None]
        cols.append(dict(name=path.rsplit('/', 1)[-1].replace('.jsonl', ''), n=len(rows), errors=len(rows) - len(ok), acc=acc,
                         back=back, real=real, overall=mean([back, real]), by_len=by_len,
                         frames=mean([r['frames'] for r in ok]), tokens=mean([r['input_tokens'] for r in ok]),
                         max_tokens=max((r['input_tokens'] for r in ok), default=0), gen_s=mean([r['generate_s'] for r in ok]),
                         peak=max((r['peak_gb'] for r in ok), default=0)))
    w = max(12, *(len(c['name']) for c in cols))
    line = lambda label, vals, fmt='%.1f': print(f'{label:<22}' + ''.join(f'{(fmt % v) if v == v else "-":>{w + 2}}' for v in vals))
    print(f'{"":<22}' + ''.join(f'{c["name"]:>{w + 2}}' for c in cols))
    print(f'{"questions (errors)":<22}' + ''.join(f'{"%d (%d)" % (c["n"], c["errors"]):>{w + 2}}' for c in cols))
    for t in BACKWARD:
        line(f'  {t}', [c['acc'].get(t, float('nan')) for c in cols])
    line('Backward avg', [c['back'] for c in cols])
    for t in REALTIME:
        line(f'  {t}', [c['acc'].get(t, float('nan')) for c in cols])
    line('Real-time avg', [c['real'] for c in cols])
    line('Overall (B+R)/2', [c['overall'] for c in cols])
    print('by clip length (query time)')
    for b in LENGTH_BINS:
        line(f'  {b[0]}-{b[1] if b[1] < 10**9 else "max"} s', [c['by_len'][b] for c in cols])
    print('cost per question')
    line('  frames (mean)', [c['frames'] for c in cols], '%.0f')
    line('  input tokens (mean)', [c['tokens'] for c in cols], '%.0f')
    line('  input tokens (max)', [c['max_tokens'] for c in cols], '%.0f')
    line('  generate s (mean)', [c['gen_s'] for c in cols], '%.2f')
    line('  peak GPU GB (max)', [c['peak'] for c in cols], '%.1f')

if __name__ == '__main__':
    main([a for a in sys.argv[1:] if a != "--common"], common="--common" in sys.argv)
