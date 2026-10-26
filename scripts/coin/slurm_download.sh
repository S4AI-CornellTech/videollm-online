#!/bin/bash
#SBATCH --job-name=coin-download
#SBATCH --partition=g3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --mail-user=xc562@cornell.edu
#SBATCH --mail-type=BEGIN,FAIL
# Step 1/3: download a stratified COIN subset (no GPU), one video at a time to avoid YouTube rate limits.
# Resumable: re-submit to continue; finished videos are kept. Stops by itself if YouTube starts blocking.
# Usage: COOKIES=$PWD/cookies.txt PER_TASK_TRAIN=10 PER_TASK_TEST=5 sbatch scripts/coin/slurm_download.sh
# SEED: download order (use a different one than a laptop downloading at the same time, so they do not chase the same videos)
# SHARD: k/n, download only the k-th of n disjoint parts (e.g. 0/2 here, 1/2 on a laptop; or 0/4 here, 2/4 on a laptop to split half 0) so they never overlap
# UPLOADED_LIST: video ids to count as done without re-verifying them with ffmpeg
cd "$SLURM_SUBMIT_DIR"
source vlo-env/bin/activate
python -m data.coin.download_videos \
    --json_path datasets/coin/coin.json \
    --output_dir datasets/coin/videos \
    --ffmpeg ffmpeg \
    --per_task_train ${PER_TASK_TRAIN:-10} \
    --per_task_test ${PER_TASK_TEST:-5} \
    --sleep ${SLEEP:-5} \
    --seed ${SEED:-0} \
    --shard ${SHARD:-0/1} \
    ${UPLOADED_LIST:+--uploaded_list $UPLOADED_LIST} \
    ${COOKIES:+--cookies $COOKIES}
