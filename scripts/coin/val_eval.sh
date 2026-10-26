#!/bin/bash
#SBATCH --job-name=coin-val
#SBATCH --partition=g3
#SBATCH --qos=gupta
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --exclude=g3-compute-17
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --mail-user=xc562@cornell.edu
#SBATCH --mail-type=END,FAIL
# Score checkpoints on the 100 held-out training videos (splits/metadata_val.json from make_splits.py), to pick the best one
# before the final test evaluation. The coin_*_train benchmarks build the same questions as *_test, here only for those 100 videos.
# Usage (from the repo root): CKPTS="<dir> <dir> ..." sbatch scripts/coin/val_eval.sh   (checkpoints already scored are skipped)
#   or RUN=<output dir> sbatch scripts/coin/val_eval.sh to score all its checkpoint-* and the final model, found when the job starts
set -e
cd "$SLURM_SUBMIT_DIR"
source vlo-env/bin/activate
export HF_HUB_OFFLINE=1
export COIN_METADATA=outputs/coin_benchmarks/splits/metadata_val.json
if [ -n "$RUN" ]; then CKPTS="$(ls -d $RUN/checkpoint-* | sort -t- -k2 -n | tr '\n' ' ') $RUN"; fi
for ckpt in ${CKPTS:?set CKPTS}; do
    log=$ckpt/val_eval.log
    if grep -q "'eval_coin_task_train_accuracy'" $log 2>/dev/null; then continue; fi
    echo "===== $ckpt"
    torchrun --nproc_per_node=1 --standalone evaluate.py \
        --live_version live1+ \
        --attn_implementation sdpa \
        --eval_datasets coin_step_train coin_next_train coin_task_train \
        --per_device_eval_batch_size 1 \
        --prediction_loss_only False \
        --dataloader_num_workers 16 \
        --bf16 True \
        --tf32 True \
        --report_to none \
        --output_dir $ckpt/val_eval \
        --resume_from_checkpoint $ckpt > $log 2>&1 \
        || { echo "!!! failed, end of $log:"; tail -30 $log; exit 1; }
done
# table over every scored checkpoint of the same runs
python - ${CKPTS} <<'PY'
import re, sys, os, glob
runs = sorted({os.path.dirname(c.rstrip('/')) if os.path.basename(c.rstrip('/')).startswith('checkpoint-') else c.rstrip('/') for c in sys.argv[1:]})
print('\n%-60s %8s %8s %8s %8s' % ('val accuracy (%), 100 videos', 'step', 'next', 'task', 'mean'))
for run in runs:
    for log in sorted(glob.glob(f'{run}/checkpoint-*/val_eval.log') + glob.glob(f'{run}/val_eval.log'), key=lambda p: int(re.findall(r'checkpoint-(\d+)', p)[0]) if 'checkpoint-' in p else 10**9):
        accs = dict(re.findall(r"'eval_coin_([a-z]+)_train_accuracy': ([0-9.]+)", open(log).read()))
        if len(accs) == 3:
            vals = [float(accs[t]) for t in ['step', 'next', 'task']]
            print('%-60s %8.1f %8.1f %8.1f %8.1f' % (os.path.dirname(log), *vals, sum(vals) / 3))
PY
