#!/bin/bash
# Run on the laptop, in the folder that has download_videos.py and videos/.
# 1. uploads finished videos to the cluster
# 2. writes uploaded.txt from what is actually on the cluster (the source of truth)
# 3. deletes local videos whose cluster copy has the same size
# Then restart downloads with --uploaded_list uploaded.txt so deleted videos are not downloaded again.
# Usage: ./sync_to_cluster.sh            (or pass another user@host, e.g. an ssh alias)
set -euo pipefail
HOST=${1:-xc562@en-cc-g3-login-01.coecis.cornell.edu}
REMOTE=videollm-online/datasets/coin/videos

# COIN ids have no dots, so '*.*.*' skips yt-dlp leftovers (id.f140.m4a, id.temp.mp4, id.mp4.part)
rsync -av --partial --exclude '*.*.*' --exclude '*.part' --include '*.mp4' --exclude '*' videos/ "$HOST:$REMOTE/"

ssh "$HOST" "cd ~/$REMOTE && find . -maxdepth 1 -name '*.mp4' ! -name '*.*.*' -printf '%f %s\n'" > uploaded_sizes.txt
cut -d' ' -f1 uploaded_sizes.txt > uploaded.txt

deleted=0
while read -r name size; do
    f="videos/$name"
    if [ -f "$f" ] && [ "$(stat -f %z "$f")" = "$size" ]; then
        rm "$f"
        deleted=$((deleted + 1))
    fi
done < uploaded_sizes.txt
echo "$(wc -l < uploaded.txt | tr -d ' ') videos on the cluster; deleted $deleted local copies; $(ls videos | grep -cE '^[^.]+\.mp4$' || true) left locally"
