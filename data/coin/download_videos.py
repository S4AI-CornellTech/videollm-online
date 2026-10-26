# standalone: needs only yt-dlp (+ deno, ffmpeg), so it can run on a laptop and the videos be rsynced to the cluster
import json, os, argparse, subprocess, random, time, shutil, sys, zlib
import concurrent.futures

def download_video(video_id, video_url, output_dir, ffmpeg_location=None, cookies=None):
    output_path = os.path.join(output_dir, f'{video_id}.mp4')
    if os.path.exists(output_path):
        try:
            ffmpeg_cmd = ["ffmpeg", "-v", "error", "-i", output_path, "-f", "null", "-"]
            if ffmpeg_location:
                ffmpeg_cmd[0] = os.path.join(ffmpeg_location, "ffmpeg")
            subprocess.run(ffmpeg_cmd, check=True)
            print(f'{output_path} has been downloaded and verified...')
            return
        except:
            print(f'{output_path} may be broken. Downloading it again...')
            os.remove(output_path)
    # frames are downscaled to 384 later, so <=480p is enough
    cmd = ["yt-dlp", "-f", "bv*[height<=480][ext=mp4]+ba[ext=m4a]/b[height<=480][ext=mp4]/b[ext=mp4]/b", "--merge-output-format", "mp4", "--sleep-requests", "1", "-o", output_path, video_url]
    if cookies:
        cmd.extend(["--cookies", cookies])
    if ffmpeg_location:
        cmd.extend(["--ffmpeg-location", ffmpeg_location])
    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        raise DownloadError(result.stderr)

# yt-dlp messages for videos we can never get (gone, private, login/members only), as opposed to YouTube blocking us
DEAD_LINK_MESSAGES = ['video is unavailable', 'Video unavailable', 'Private video', 'has been removed', 'account associated with this video has been terminated',
                      'confirm your age', 'Please sign in', 'members', 'Join this channel', 'Requested format is not available']
BLOCK_MESSAGES = ['not a bot', 'try again later', 'rate-limit', 'rate limit', 'HTTP Error 429']

class DownloadError(Exception):
    def __init__(self, stderr):
        super().__init__(stderr.strip().splitlines()[-1] if stderr.strip() else 'yt-dlp failed')
        self.dead_link = any(m in stderr for m in DEAD_LINK_MESSAGES) and not any(m in stderr for m in BLOCK_MESSAGES)

def main_subset(output_dir, json_path, ffmpeg_location, cookies, per_task_train, per_task_test, seed, sleep, max_consecutive_failures, uploaded=frozenset(), shard=(0, 1)):
    # stratified subset: per_task_train/per_task_test available videos for each of the 180 tasks.
    # one video at a time, round-robin over tasks, so stopping early still leaves a balanced subset.
    annotations = json.load(open(json_path, 'r'))['database']
    rng = random.Random(seed)
    candidates, targets = {}, {'training': per_task_train, 'testing': per_task_test}
    for video_id, anno in sorted(annotations.items()):
        if zlib.crc32(video_id.encode()) % shard[1] != shard[0]: # disjoint halves for machines downloading in parallel
            continue
        candidates.setdefault((anno['class'], anno['subset']), []).append((video_id, anno['video_url']))
    for videos in candidates.values():
        rng.shuffle(videos) # dead links are skipped, and the next candidate of the same task is tried
    ready = {key: 0 for key in candidates}
    failed, consecutive_failures = [], 0
    while True:
        pending = [key for key in sorted(candidates) if ready[key] < targets[key[1]] and candidates[key]]
        if not pending:
            break
        for key in pending:
            video_id, video_url = candidates[key].pop(0)
            if video_id in uploaded: # already on the cluster and deleted locally: counts as done
                ready[key] += 1
                continue
            already = os.path.exists(os.path.join(output_dir, f'{video_id}.mp4'))
            try:
                download_video(video_id, video_url, output_dir, ffmpeg_location, cookies)
                ready[key] += 1
                consecutive_failures = 0
            except Exception as e:
                print(f'failed {video_id}: {e}')
                failed.append(video_id)
                # ~1/4 of COIN links are dead, and dead links can come in long runs, so only other failures count towards stopping
                if not getattr(e, 'dead_link', False):
                    consecutive_failures += 1
                    print(f'  ^ possible block, counts towards stopping: {consecutive_failures}/{max_consecutive_failures}')
                if consecutive_failures >= max_consecutive_failures:
                    print(f'!!! {consecutive_failures} failures in a row that are not dead links: YouTube is likely rate-limiting this account/IP. '
                          'Stopping; wait a few hours, refresh cookies, then resubmit (finished videos are kept).')
                    _report(ready, targets, failed, output_dir)
                    raise SystemExit(1)
            if not already:
                time.sleep(rng.uniform(sleep, 2 * sleep))
    _report(ready, targets, failed, output_dir)

def _report(ready, targets, failed, output_dir):
    for subset in ['training', 'testing']:
        counts = [n for (_, s), n in ready.items() if s == subset]
        short = sum(n < targets[subset] for n in counts)
        print(f'{subset}: {sum(counts)} videos ready, {short}/{len(counts)} tasks below {targets[subset]} per task')
    with open(os.path.join(output_dir, '..', 'failed_downloads.txt'), 'a') as f:
        f.write(''.join(f'{video_id}\n' for video_id in failed))

def main(output_dir, json_path, num_workers, ffmpeg_location, cookies):
    annotations = json.load(open(json_path, 'r'))['database']
    annotations = list(annotations.items())
    random.shuffle(annotations)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(download_video, video_id, annotation['video_url'], output_dir, ffmpeg_location, cookies) for video_id, annotation in annotations]
        concurrent.futures.wait(futures)
    failed = [video_id for (video_id, _), future in zip(annotations, futures) if future.exception() is not None]
    print(f'{len(annotations) - len(failed)}/{len(annotations)} videos ready, {len(failed)} failed (often unavailable on YouTube)')
    with open(os.path.join(output_dir, '..', 'failed_downloads.txt'), 'w') as f:
        f.write('\n'.join(failed))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download videos in parallel using yt-dlp')
    parser.add_argument('--output_dir', type=str, default='datasets/coin/videos', help='Directory to save downloaded videos')
    parser.add_argument('--json_path', type=str, default='datasets/coin/coin.json', help='Path to the JSON file containing video data')
    parser.add_argument('--ffmpeg', type=str, default=None)
    parser.add_argument('--cookies', type=str, default=None, help='Netscape cookies.txt for YouTube, if downloads get bot-checked')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of parallel downloads')
    parser.add_argument('--per_task_train', type=int, default=None, help='Download a stratified subset: this many training videos per task (sequential, throttled)')
    parser.add_argument('--per_task_test', type=int, default=None, help='With --per_task_train: this many testing videos per task')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sleep', type=float, default=5, help='Subset mode: sleep uniform(sleep, 2*sleep) seconds after each new download')
    parser.add_argument('--max_consecutive_failures', type=int, default=10, help='Subset mode: stop when this many downloads fail in a row')
    parser.add_argument('--shard', type=str, default='0/1', help='Subset mode: k/n downloads only the k-th of n disjoint parts of COIN (e.g. 0/2 on the cluster, 1/2 on a laptop)')
    parser.add_argument('--uploaded_list', type=str, default=None, help='Subset mode: file of video ids already on the cluster (one per line, .mp4 optional); skipped even if deleted locally')

    args = parser.parse_args()

    for tool in ['yt-dlp', 'deno']:
        if shutil.which(tool) is None:
            sys.exit(f'{tool} not found on PATH. Activate the venv where you ran `pip install "yt-dlp[default]"`, and install deno.')
    os.makedirs(args.output_dir, exist_ok=True)
    if args.per_task_train is not None:
        uploaded = set()
        if args.uploaded_list and os.path.exists(args.uploaded_list):
            uploaded = {os.path.splitext(line.strip())[0] for line in open(args.uploaded_list) if line.strip()}
            print(f'{len(uploaded)} videos already uploaded, skipping them')
        main_subset(args.output_dir, args.json_path, args.ffmpeg, args.cookies, args.per_task_train, args.per_task_test or 0,
                    args.seed, args.sleep, args.max_consecutive_failures, uploaded, tuple(map(int, args.shard.split('/'))))
    else:
        main(args.output_dir, args.json_path, args.num_workers, args.ffmpeg, args.cookies)