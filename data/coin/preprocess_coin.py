"""Two-stage COIN preprocessing: ffmpeg to 2fps/384, then SigLIP frame embeddings.

Single-process replacement for data/preprocess/{ffmpeg,encode}.py, which fan out
over 8 GPUs / SLURM via submitit. It also fixes two things that break here:

  - distributed_ffmpeg writes `videos_2fps_max384`, but distributed_encode and
    the COIN dataset class both expect `videos_2fps_384`.
  - distributed_encode calls torchvision.io.read_video, removed in torchvision
    0.29 (the same breakage already patched in demo/inference.py).

Both stages skip outputs that already exist, so runs are resumable and a subset
run is a strict prefix of the full run.

python -m data.coin.preprocess_coin --stage all
"""
import argparse, json, os, av, numpy as np, torch, tqdm
from dataclasses import asdict

from models.arguments_live import LiveOnePlusTrainingArguments
from models.configuration_live import LiveConfigMixin
from models.vision_live import build_live_vision
from ..utils import ffmpeg_once

def read_video_frames(path: str) -> torch.Tensor:
    """T x C x H x W uint8, via PyAV (torchvision.io.read_video no longer exists)."""
    container = av.open(path)
    frames = [f.to_ndarray(format='rgb24') for f in container.decode(video=0)]
    container.close()
    if not frames:
        raise ValueError(f'no decodable frames in {path}')
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()

def stage_ffmpeg(video_dir: str, fps: int, resolution: int):
    dst_dir = f'{video_dir}_{fps}fps_{resolution}'
    os.makedirs(dst_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(video_dir) if n.endswith('.mp4'))
    failed = {}
    for name in tqdm.tqdm(names, desc=f'ffmpeg -> {dst_dir}'):
        dst_path = os.path.join(dst_dir, name)
        if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
            continue
        try:
            ffmpeg_once(os.path.join(video_dir, name), dst_path, fps=fps, resolution=resolution)
        except Exception as error:
            failed[name] = str(error)
            if os.path.exists(dst_path):
                os.remove(dst_path)
    print(f'ffmpeg: {len(names) - len(failed)} ok, {len(failed)} failed')
    return dst_dir, failed

def stage_encode(frame_dir: str, args, batch_size: int):
    vision_config = LiveConfigMixin(**asdict(args))
    vision_model, vision_encode = build_live_vision(vision_config)
    vision_model = vision_model.to('cuda').eval()

    # must match COIN.embed_dir: f"{video_root}_{embed_mark}_{vision_pretrained/->--}"
    dst_dir = f"{frame_dir}_{args.embed_mark.split('_')[-1]}_{args.vision_pretrained.replace('/', '--')}"
    os.makedirs(dst_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(frame_dir) if n.endswith('.mp4'))
    failed = {}
    for name in tqdm.tqdm(names, desc=f'encode -> {dst_dir}'):
        save_path = os.path.join(dst_dir, os.path.splitext(name)[0] + '.pt')
        if os.path.exists(save_path):
            continue
        try:
            frames = read_video_frames(os.path.join(frame_dir, name))
            with torch.no_grad():
                embeds = torch.cat([
                    vision_encode(vision_model, batch.to('cuda')).cpu()
                    for batch in frames.split(batch_size)
                ])
            torch.save(embeds.to(torch.bfloat16), save_path)
        except Exception as error:
            failed[name] = str(error)
            if os.path.exists(save_path):
                os.remove(save_path)
    print(f'encode: {len(names) - len(failed)} ok, {len(failed)} failed')
    return dst_dir, failed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', default='datasets/coin/videos')
    parser.add_argument('--stage', default='all', choices=['ffmpeg', 'encode', 'all'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--frame_fps', type=int, default=None,
                        help='override the training default (2). The demo runs inference at 10; '
                             'evaluate with a matching --embed_mark <fps>fps_384_1+3x3.')
    parser.add_argument('--report', default=None)
    cli_args = parser.parse_args()

    args = LiveOnePlusTrainingArguments(output_dir='outputs/preprocess')
    if cli_args.frame_fps is not None:
        args.frame_fps = cli_args.frame_fps
    # embed_mark's trailing token (1+3x3) is fps-independent, so the fps only
    # enters through frame_dir -> videos_<fps>fps_384_1+3x3_<vision_pretrained>
    frame_dir = f'{cli_args.video_dir}_{args.frame_fps}fps_{args.frame_resolution}'
    if cli_args.report is None:
        cli_args.report = f'datasets/coin/preprocess_report_{args.frame_fps}fps.json'
    report = {}

    if cli_args.stage in ('ffmpeg', 'all'):
        frame_dir, report['ffmpeg_failed'] = stage_ffmpeg(cli_args.video_dir, args.frame_fps, args.frame_resolution)
    if cli_args.stage in ('encode', 'all'):
        embed_dir, report['encode_failed'] = stage_encode(frame_dir, args, cli_args.batch_size)
        report['embed_dir'] = embed_dir
        print(f'\nembeddings -> {embed_dir}')

    json.dump(report, open(cli_args.report, 'w'), indent=2)

if __name__ == '__main__':
    main()
