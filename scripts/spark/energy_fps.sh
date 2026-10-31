#!/bin/bash
# Energy of streaming VideoLLM-online at different frame rates on the DGX Spark: every clip in CLIP_DIR at every FPS,
# one demo.cli process per (fps, clip), with power logged throughout by profiling_scripts/power_log.py.
# It starts with IDLE_S seconds of idle power (the baseline) and pauses COOL_S seconds between runs.
# Close other GPU work first: the power log covers the whole machine's GPU.
# Usage (from the repo root, conda env videollm):
#   bash scripts/spark/energy_fps.sh
#   FPS_LIST="0.5 1 2 4" CLIP_DIR=outputs/profiling/energy_clips OUT=outputs/profiling/energy bash scripts/spark/energy_fps.sh
# then: python -m profiling_scripts.energy_report outputs/profiling/energy
set -e
FPS_LIST=${FPS_LIST:-0.5 1 2 4}
CLIP_DIR=${CLIP_DIR:-outputs/profiling/energy_clips}
OUT=${OUT:-outputs/profiling/energy}
CKPT=${CKPT:-checkpoints/coin-ft}
IDLE_S=${IDLE_S:-60}
COOL_S=${COOL_S:-20}
mkdir -p $OUT
ls $CLIP_DIR/*.mp4 > /dev/null

python -m profiling_scripts.power_log --probe | tee $OUT/power_sources.txt
python -m profiling_scripts.power_log $OUT/power.csv &
LOGGER=$!
trap 'kill $LOGGER 2>/dev/null' EXIT
echo "idle baseline: ${IDLE_S}s"
echo "{\"idle_start\": $(date +%s.%N), \"idle_end\": $(sleep $IDLE_S; date +%s.%N)}" > $OUT/idle.json

for FPS in $FPS_LIST; do
    for CLIP in $CLIP_DIR/*.mp4; do
        NAME=$(basename $CLIP .mp4)_${FPS}fps
        [ -f $OUT/$NAME.json ] && { echo "skip $NAME (done)"; continue; } # rerunning the script resumes
        echo "===== $NAME"
        VIDEO=$CLIP OUT_PREFIX=$OUT/$NAME python -m demo.cli --resume_from_checkpoint $CKPT --attn_implementation sdpa_gqa \
            --kv_prealloc -1 --frame_fps $FPS --max_reply_tokens 32 --phase_timing True > $OUT/$NAME.log 2>&1 \
            || { echo "!!! $NAME failed, end of $OUT/$NAME.log:"; tail -20 $OUT/$NAME.log; }
        rm -f $OUT/${NAME}_phase_timeline.png
        sleep $COOL_S
    done
done
kill $LOGGER; wait $LOGGER 2>/dev/null || true
echo "done: python -m profiling_scripts.energy_report $OUT"
