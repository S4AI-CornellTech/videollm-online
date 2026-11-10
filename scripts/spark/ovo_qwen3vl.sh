#!/bin/bash
# OVO-Bench subset with Qwen3-VL on the DGX Spark at several frame rates, with power logged throughout
# (profiling_scripts/power_log.py), so every question gets its time, energy and peak memory on the edge device.
# Same evaluator as the B300 runs (scripts/ovo/qwen3vl_fps_eval.py); without decord it reads frames with PyAV.
# It starts with IDLE_S seconds of idle power (the baseline) and pauses COOL_S seconds between frame rates.
# Close other GPU work first: the power log covers the whole machine.
# Setup on the Spark (once):
#   conda create -n qwen3vl --clone videollm && conda activate qwen3vl && pip install -U "transformers==4.57.6" accelerate av
#   huggingface-cli download Qwen/Qwen3-VL-8B-Instruct
#   copy OVO-Bench's ovo_bench_new.json and the subset's videos (scripts/ovo/spark_subset_videos.txt) to OVO_DIR
# Usage (from the repo root, conda env qwen3vl):
#   bash scripts/spark/ovo_qwen3vl.sh
#   FPS_LIST="0.5 1 2" OVO_DIR=~/ovo OUT=outputs/profiling/spark_ovo bash scripts/spark/ovo_qwen3vl.sh
# then: python -m profiling_scripts.ovo_energy_report outputs/profiling/spark_ovo
set -e
FPS_LIST=${FPS_LIST:-0.5 1 2}
OVO_DIR=${OVO_DIR:-$HOME/ovo}
OUT=${OUT:-outputs/profiling/spark_ovo}
IDS=${IDS:-scripts/ovo/spark_subset.txt}
MAX_TOKENS=${MAX_TOKENS:-131072} # keeps peak memory near 60 GB of the 128 GB shared with the OS
IDLE_S=${IDLE_S:-60}
COOL_S=${COOL_S:-60}
mkdir -p $OUT
ls $OVO_DIR/ovo_bench_new.json > /dev/null

python -m profiling_scripts.power_log --probe | tee $OUT/power_sources.txt
python -m profiling_scripts.power_log $OUT/power.csv &
LOGGER=$!
trap 'kill $LOGGER 2>/dev/null' EXIT
until [ "$(wc -l < $OUT/power.csv 2>/dev/null || echo 0)" -gt 2 ]; do sleep 1; done # the logger can take a while to start
echo "idle baseline: ${IDLE_S}s"
echo "{\"idle_start\": $(date +%s.%N), \"idle_end\": $(sleep $IDLE_S; date +%s.%N)}" > $OUT/idle.json

for FPS in $FPS_LIST; do # rerunning resumes: finished questions are skipped
    echo "===== ${FPS} fps"
    python scripts/ovo/qwen3vl_fps_eval.py --fps $FPS --out $OUT/qwen3vl8b_${FPS}fps.jsonl --ids $IDS \
        --anno $OVO_DIR/ovo_bench_new.json --video_dir $OVO_DIR/src_videos --max_input_tokens $MAX_TOKENS 2>&1 | tee -a $OUT/qwen3vl8b_${FPS}fps.log
    sleep $COOL_S
done
kill $LOGGER; wait $LOGGER 2>/dev/null || true
echo "done: python -m profiling_scripts.ovo_energy_report $OUT"
