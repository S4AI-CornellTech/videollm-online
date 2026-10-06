"""Download COIN videos for a given split.

Replaces download_videos.py, whose `--username oauth2 --password ''` flags were
removed from yt-dlp. Selection is deterministic (sorted video ids) so that a
--limit subset is stable across the download / ffmpeg / encode / evaluate steps.

Only the longest side matters downstream (frames are scaled to 384 and padded)
and audio is dropped by ffmpeg_once's -an, so we fetch video-only streams capped
at 480p to save bandwidth.

python -m data.coin.download_coin --split testing --limit 50
"""
import argparse, json, os, subprocess, sys
import concurrent.futures

FORMAT = (
    'bv*[height<=?480][ext=mp4]/bv*[height<=?480]/'
    'b[height<=?480][ext=mp4]/b[height<=?480]/bv*/b'
)

def select_videos(json_path: str, split: str, limit: int | None):
    database = json.load(open(json_path))['database']
    video_ids = sorted(vid for vid, anno in database.items() if split in anno['subset'].lower())
    return video_ids[:limit] if limit is not None else video_ids

def download_video(video_id: str, output_dir: str):
    output_path = os.path.join(output_dir, f'{video_id}.mp4')
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return video_id, True, 'cached'
    cmd = [
        sys.executable, '-m', 'yt_dlp',
        '-f', FORMAT,
        '--remux-video', 'mp4',
        '--no-playlist', '--no-progress', '--quiet', '--no-warnings',
        '--retries', '3', '--socket-timeout', '30',
        '-o', os.path.join(output_dir, f'{video_id}.%(ext)s'),
        f'https://www.youtube.com/watch?v={video_id}',
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0 and os.path.exists(output_path):
        return video_id, True, 'downloaded'
    message = (proc.stderr or proc.stdout).strip().splitlines()
    return video_id, False, message[-1] if message else 'unknown error'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', default='datasets/coin/coin.json')
    parser.add_argument('--output_dir', default='datasets/coin/videos')
    parser.add_argument('--split', default='testing', choices=['training', 'testing'])
    parser.add_argument('--limit', type=int, default=None, help='first N ids (sorted) for a reproducible subset')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--manifest', default='datasets/coin/download_manifest.json')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    video_ids = select_videos(args.json_path, args.split, args.limit)
    print(f'{len(video_ids)} videos in split={args.split}' + (f' (limited to {args.limit})' if args.limit else ''), flush=True)

    ok, failed, done = [], {}, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(download_video, vid, args.output_dir) for vid in video_ids]
        for future in concurrent.futures.as_completed(futures):
            video_id, success, detail = future.result()
            done += 1
            (ok.append(video_id) if success else failed.update({video_id: detail}))
            if done % 25 == 0 or done == len(video_ids):
                print(f'  {done}/{len(video_ids)}  ok={len(ok)}  failed={len(failed)}', flush=True)

    manifest = {'split': args.split, 'requested': len(video_ids), 'ok': sorted(ok), 'failed': failed}
    if os.path.exists(args.manifest):  # merge with a previous run so subset -> full stays additive
        previous = json.load(open(args.manifest))
        if previous.get('split') == args.split:
            merged_ok = set(previous.get('ok', [])) | set(ok)
            manifest['ok'] = sorted(merged_ok)
            manifest['failed'] = {k: v for k, v in {**previous.get('failed', {}), **failed}.items() if k not in merged_ok}
    json.dump(manifest, open(args.manifest, 'w'), indent=2)
    print(f"\n{len(manifest['ok'])} available, {len(manifest['failed'])} unavailable -> {args.manifest}")

if __name__ == '__main__':
    main()
