# Print the checkpoint of a run with the best mean validation accuracy (step, next, task), from the val_eval.log files val_eval.sh writes.
# Usage: vlo-env/bin/python scripts/coin/pick_best.py <run output dir>
import re, sys, glob, os

run = sys.argv[1].rstrip('/')
scores = {}
for log in glob.glob(f'{run}/checkpoint-*/val_eval.log') + glob.glob(f'{run}/val_eval.log'):
    accs = dict(re.findall(r"'eval_coin_([a-z]+)_train_accuracy': ([0-9.]+)", open(log).read()))
    if len(accs) == 3:
        scores[os.path.dirname(log)] = sum(float(a) for a in accs.values()) / 3
if not scores:
    sys.exit(f'no validation scores under {run}')
best = max(scores, key=scores.get)
print(f'best on validation: {best} ({scores[best]:.1f} mean)', file=sys.stderr)
print(best)
