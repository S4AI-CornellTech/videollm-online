#!/bin/bash
#SBATCH --job-name=coin-compare
#SBATCH --partition=g3
#SBATCH --qos=gupta
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --mail-user=xc562@cornell.edu
#SBATCH --mail-type=BEGIN,END,FAIL
# Evaluate the original v1plus and the fine-tuned model on the five COIN test benchmarks,
# on one frozen snapshot of the encoded videos, so both see exactly the same test set.
# Usage: CKPT=outputs/coin_benchmarks/live1+_from_v1plus sbatch scripts/coin/compare_eval.sh
# Reuse an earlier baseline instead of re-running v1plus (it is deterministic on the same videos):
#   SNAPSHOT=<metadata json that run used> BASELINE_LOG=<its slurm log> sbatch scripts/coin/compare_eval.sh
set -e
cd "$SLURM_SUBMIT_DIR"
source vlo-env/bin/activate
CKPT=${CKPT:-outputs/coin_benchmarks/live1+_from_v1plus}
OUT=outputs/coin_benchmarks/compare_$SLURM_JOB_ID
mkdir -p $OUT
cp "${SNAPSHOT:-datasets/coin/videos_2fps_384_1+3x3_google--siglip-large-patch16-384_metadata.json}" $OUT/metadata_snapshot.json
export COIN_METADATA=$OUT/metadata_snapshot.json
echo "test-set snapshot: $(python -c "import json,sys; m=json.load(open(sys.argv[1])); d=json.load(open('datasets/coin/coin.json'))['database']; print(sum(d[k]['subset']=='testing' for k in m if k in d))" $COIN_METADATA) test videos"
MODELS=(chenjoya/videollm-online-8b-v1plus "$CKPT")
if [ -n "$BASELINE_LOG" ]; then
    tr '\r' '\n' < "$BASELINE_LOG" > $OUT/videollm-online-8b-v1plus.log # v1plus results from the earlier run
    MODELS=("$CKPT")
fi
for model in "${MODELS[@]}"; do
    name=$(basename "$model")
    echo "===== evaluating $model"
    torchrun --nproc_per_node=${SLURM_GPUS_ON_NODE:-1} --standalone evaluate.py \
        --live_version live1+ \
        --attn_implementation sdpa \
        --eval_datasets coin_step_test coin_next_test coin_task_test coin_procedure_test coin_taskprocedure_test \
        --per_device_eval_batch_size 1 \
        --prediction_loss_only False \
        --dataloader_num_workers 16 \
        --bf16 True \
        --tf32 True \
        --report_to none \
        --output_dir $OUT/$name \
        --resume_from_checkpoint "$model" > $OUT/$name.log 2>&1 \
        || { echo "!!! evaluation of $model failed, end of $OUT/$name.log:"; tail -30 $OUT/$name.log; exit 1; }
    grep -E "^\{.*eval_" $OUT/$name.log
done
python - $OUT <<'PY'
import re, sys, os
out = sys.argv[1]
rows = {}
for log in sorted(os.listdir(out)):
    if log.endswith('.log'):
        text = open(os.path.join(out, log)).read()
        rows[log[:-4]] = dict(re.findall(r"'eval_coin_([a-z]+)_test_accuracy': ([0-9.]+)", text))
tasks = ['step', 'next', 'task', 'procedure', 'taskprocedure']
print('\n%-32s' % 'accuracy (%)' + ''.join('%15s' % t for t in tasks))
for name, accs in rows.items():
    print('%-32s' % name + ''.join('%15s' % (f"{float(accs[t]):.1f}" if t in accs else '-') for t in tasks))
PY
