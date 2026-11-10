# OVO-Bench backward-tracing + real-time perception with Qwen3-VL at a fixed frame rate (FPS vs accuracy study).
# Unlike the official offline protocol (up to 64 frames spread uniformly over the clip), every question gets frames at
# exactly t = 0, 1/fps, 2/fps, ... up to its query time (the official chunker's ceil(realtime) cut), each frame at the same
# fixed resolution, so only the frame rate changes between runs. Frames are decoded straight from the source videos, and
# Qwen3-VL's own frame sampling / resizing are switched off (its defaults would cap a clip at 768 frames and shrink frames).
# One JSON line per question (response, frames, visual tokens, timings, peak memory); rerunning resumes.
# Run with the Qwen3-VL env (transformers >= 4.57), e.g.
#   python scripts/ovo/qwen3vl_fps_eval.py --fps 1 --out outputs/ovo/qwen3vl8b_1fps.jsonl [--limit 20] [--shard 0/2]
# Score: python scripts/ovo/score_fps.py outputs/ovo/qwen3vl8b_*fps.jsonl
import argparse, json, math, os, sys, time
import numpy as np, torch
try:
    import decord
except ImportError: # e.g. aarch64 (DGX Spark), where decord has no wheel: frames are read with PyAV instead
    decord = None
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.video_utils import VideoMetadata

BACKWARD_TASKS = ['EPM', 'ASI', 'HLD']
REAL_TIME_TASKS = ['OCR', 'ACR', 'ATR', 'STU', 'FPD', 'OJR']
# OVO-Bench's prompt for these tasks (constant.py, BR_PROMPT_TEMPLATE)
PROMPT = """
Question: {}
Options:
{}

Respond only with the letter corresponding to your chosen option (e.g., A, B, C).
Do not include any additional text or explanation in your response.
"""
FACTOR = 32 # Qwen3-VL: 16-pixel patches, 2x2 merge

class StageTimer:
    """Splits model.generate into vision encode, prefill (first forward minus vision) and decode (later forwards), timed with
    forward hooks and CUDA syncs around the whole model and its vision tower. The syncs cost a little time, so it is opt-in."""
    def __init__(self, model):
        self.calls, self.vision = [], []
        for mod, store in ((model, self.calls), (model.model.visual, self.vision)):
            mod.register_forward_pre_hook(lambda m, a, store=store: self._start(store))
            mod.register_forward_hook(lambda m, a, o, store=store: self._stop(store))

    def _start(self, store):
        torch.cuda.synchronize(); store.append(-time.perf_counter())

    def _stop(self, store):
        torch.cuda.synchronize(); store[-1] += time.perf_counter()

    def reset(self):
        self.calls.clear(); self.vision.clear()

    def result(self):
        vision = sum(self.vision)
        return dict(vision_s=round(vision, 4), prefill_s=round(self.calls[0] - vision, 4), decode_s=round(sum(self.calls[1:]), 4),
                    decode_steps=len(self.calls) - 1)

def frame_size(h, w, pixels):
    """Height and width with about `pixels` pixels, the source aspect ratio, and both sides multiples of FACTOR."""
    s = math.sqrt(pixels / (h * w))
    return max(FACTOR, round(h * s / FACTOR) * FACTOR), max(FACTOR, round(w * s / FACTOR) * FACTOR)

def probe(path):
    """Frame count, average fps, height and width of a video, from decord or (without it) PyAV's stream metadata."""
    if decord is not None:
        vr = decord.VideoReader(path)
        return len(vr), vr.get_avg_fps(), *vr[0].shape[:2]
    import av
    with av.open(path) as c:
        s = c.streams.video[0]
        vfps = float(s.average_rate)
        n = s.frames or int(float(s.duration * s.time_base) * vfps) # some containers do not store a frame count
        return n, vfps, s.codec_context.height, s.codec_context.width

def read_frames(path, idx, H, W):
    """Frames idx (may repeat) at H x W, as T, C, H, W uint8. PyAV decodes in order and keeps the frames it needs."""
    if decord is not None:
        vr = decord.VideoReader(path, width=W, height=H) # decode at the target size: thousands of full-size frames would not fit
        return torch.from_numpy(vr.get_batch(idx).asnumpy()).permute(0, 3, 1, 2)
    import av
    want, last, got = set(idx), max(idx), {}
    with av.open(path) as c:
        s = c.streams.video[0]; s.thread_type = 'AUTO'
        for k, fr in enumerate(c.decode(s)):
            if k in want:
                got[k] = fr.to_ndarray(format='rgb24', width=W, height=H)
            if k >= last:
                break
    if not got:
        raise RuntimeError(f'no frames decoded from {path}')
    kmax = max(got) # a stored frame count can exceed what decodes: past the end, repeat the last decoded frame
    frames = np.stack([got[min(k, kmax)] if min(k, kmax) in got else got[kmax] for k in idx])
    return torch.from_numpy(frames).permute(0, 3, 1, 2)

def estimate_tokens(path, end_s, fps, pixels):
    """Prompt length without decoding: per 2-frame group, (H/32)*(W/32) vision tokens + ~10 for timestamp and markers."""
    n, vfps, h, w = probe(path)
    H, W = frame_size(h, w, pixels)
    frames = int(math.floor(min(end_s, (n - 1) / vfps) * fps + 1e-6)) + 1
    return (frames + 1) // 2 * ((H // 32) * (W // 32) + 10)

def load_frames(path, end_s, fps, pixels):
    n, vfps, h, w = probe(path)
    H, W = frame_size(h, w, pixels)
    end_s = min(end_s, (n - 1) / vfps)
    times = [k / fps for k in range(int(math.floor(end_s * fps + 1e-6)) + 1)]
    idx = [min(round(t * vfps), n - 1) for t in times]
    if len(idx) % 2: # Qwen3-VL groups frames in pairs; repeat the last frame rather than drop it
        idx.append(idx[-1])
    frames = read_frames(path, idx, H, W) # T, C, H, W uint8
    meta = VideoMetadata(total_num_frames=n, fps=vfps, width=W, height=H, duration=n / vfps, frames_indices=idx)
    return frames, meta

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fps', type=float, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--model_path', default='Qwen/Qwen3-VL-8B-Instruct')
    p.add_argument('--anno', default='/home/xc562/ovo/OVO-Bench/data/ovo_bench_new.json')
    p.add_argument('--video_dir', default='/home/xc562/ovo/OVO-Bench/data/src_videos')
    p.add_argument('--tasks', default=','.join(BACKWARD_TASKS + REAL_TIME_TASKS))
    p.add_argument('--pixels', type=int, default=360 * 420, help='pixels per frame (OVO-Bench uses max_pixels 360*420 for Qwen2-VL)')
    p.add_argument('--max_new_tokens', type=int, default=16)
    p.add_argument('--limit', type=int, default=0, help='only the first N questions (smoke test)')
    p.add_argument('--stage_timing', action='store_true', help='log vision / prefill / decode times per question (adds CUDA syncs)')
    p.add_argument('--ids', default=None, help='file with one question id per line: run only those (e.g. a Spark subset)')
    p.add_argument('--shard', default='0/1', help='i/n: every n-th question starting at i')
    p.add_argument('--max_input_tokens', type=int, default=0,
                   help='skip questions whose prompt is longer (e.g. 262144, the trained context); logged with skipped=over_context')
    p.add_argument('--longest_first', action='store_true', help='start with the longest clips (finds memory limits early)')
    args = p.parse_args()

    tasks = set(args.tasks.split(','))
    anno = [a for a in json.load(open(args.anno)) if a['task'] in tasks]
    if args.ids:
        keep = {int(l) for l in open(args.ids) if l.strip()}
        anno = [a for a in anno if a['id'] in keep]
    if args.longest_first:
        anno.sort(key=lambda a: -a['realtime'])
    si, sn = map(int, args.shard.split('/'))
    anno = anno[si::sn]
    if args.limit:
        anno = anno[:args.limit]
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)['id'] for l in open(args.out) if l.strip()}
    todo = [a for a in anno if a['id'] not in done]
    print(f'{len(anno)} questions at {args.fps:g} fps, {len(done & {a["id"] for a in anno})} already done, {len(todo)} to go', flush=True)
    if not todo:
        return

    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    stages = StageTimer(model) if args.stage_timing else None
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'a') as f:
        for i, a in enumerate(todo):
            t0 = time.perf_counter()
            row = dict(id=a['id'], task=a['task'], video=a['video'], realtime=a['realtime'], fps=args.fps, ground_truth=chr(65 + a['gt']),
                       t_start=time.time()) # wall clock, to match against a power log
            try:
                if args.max_input_tokens: # skip clearly-too-long clips before decoding them: at high fps their frames do not fit in host memory
                    est = estimate_tokens(os.path.join(args.video_dir, a['video']), math.ceil(a['realtime']), args.fps, args.pixels)
                    if est > args.max_input_tokens + 2000:
                        row.update(response=None, skipped='over_context', input_tokens_est=est)
                        f.write(json.dumps(row) + '\n'); f.flush()
                        continue
                frames, meta = load_frames(os.path.join(args.video_dir, a['video']), math.ceil(a['realtime']), args.fps, args.pixels)
                options = '; '.join(f'{chr(65 + j)}. {o}' for j, o in enumerate(a['options'])) + ';'
                messages = [{'role': 'user', 'content': [{'type': 'video'}, {'type': 'text', 'text': PROMPT.format(a['question'], options)}]}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], videos=[frames], video_metadata=[meta], do_sample_frames=False, do_resize=False,
                                   return_tensors='pt').to('cuda')
                t1 = time.perf_counter()
                n_in = int(inputs['input_ids'].shape[1])
                if args.max_input_tokens and n_in > args.max_input_tokens:
                    row.update(response=None, skipped='over_context', frames=len(meta.frames_indices), input_tokens=n_in)
                    f.write(json.dumps(row) + '\n'); f.flush()
                    print(f'[{i + 1}/{len(todo)}] id {a["id"]} skipped: {n_in} tokens > {args.max_input_tokens}', flush=True)
                    continue
                torch.cuda.reset_peak_memory_stats()
                if stages:
                    stages.reset()
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                torch.cuda.synchronize()
                resp = processor.batch_decode(out[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
                grid = inputs['video_grid_thw'][0].tolist()
                row.update(response=resp, frames=len(meta.frames_indices), frame_hw=[meta.height, meta.width],
                           visual_tokens=int(grid[0] * grid[1] * grid[2] // 4), input_tokens=int(inputs['input_ids'].shape[1]),
                           load_s=round(t1 - t0, 3), generate_s=round(time.perf_counter() - t1, 3),
                           peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
                if stages:
                    row.update(stages.result())
            except Exception as e: # keep going; the scorer counts these as wrong, like OVO-Bench's own loop
                row.update(response=None, error=f'{type(e).__name__}: {e}'[:500])
                if isinstance(e, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
            row['t_end'] = time.time()
            f.write(json.dumps(row) + '\n'); f.flush()
            if i % 20 == 0 or row.get('error'):
                print(f'[{i + 1}/{len(todo)}] id {a["id"]} {a["task"]} {a["realtime"]:.0f}s: {row.get("frames")} frames, '
                      f'{row.get("input_tokens")} tokens, {row.get("generate_s")} s, {row.get("peak_gb")} GB -> {row.get("response")!r} '
                      f'(gt {row["ground_truth"]}){" ERROR " + row["error"] if row.get("error") else ""}', flush=True)

if __name__ == '__main__':
    main()
