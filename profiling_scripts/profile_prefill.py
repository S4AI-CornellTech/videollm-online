# Why does per-frame prefill get slower as the KV cache grows? Fills the KV cache quickly by prefilling the video's frames
# in chunks, then times one-frame prefills (11 tokens, as in demo.cli streaming) at several cache lengths, and at the longest:
#   1. which SDPA attention kernels work and how fast each is,
#   2. a torch.profiler table of the GPU / CPU ops taking the time (and a Chrome trace),
#   3. the same after merging the LoRA adapter into the base weights.
# Usage (from the repo root, same model args as demo.cli):
#   VIDEO=outputs/profiling/coin_long_20min.mp4 python -m profiling_scripts.profile_prefill \
#       --resume_from_checkpoint <adapter dir> --attn_implementation sdpa
# Optional: KV_TARGETS=2000,8000,16000,24000,28000 (tokens) STEPS=20 (timed frames per point) OUT=outputs/profiling/prefill
import os, sys, time, json
import torch
from demo.inference import LiveInfer

KV_TARGETS = [int(x) for x in os.environ.get('KV_TARGETS', '2000,8000,16000,24000,28000').split(',')]
STEPS = int(os.environ.get('STEPS', 20))
OUT = os.environ.get('OUT', 'outputs/profiling/prefill')
CHUNK_FRAMES = 64 # frames per chunk while filling the cache (704 tokens): fast, and the logits stay small
os.makedirs(OUT, exist_ok=True)

def kv_tokens(pkv):
    return pkv.get_seq_length() if hasattr(pkv, 'get_seq_length') else pkv[0][0].shape[-2]

def main():
    live = LiveInfer()
    model = live.model
    video = os.environ.get('VIDEO', 'outputs/profiling/coin_long_20min.mp4')
    fps, res = live.frame_fps, live.frame_resolution
    name, ext = os.path.splitext(video)
    cached = os.path.join('demo/assets/cache', name + f'_{fps}fps_{res}' + ext)
    if not os.path.exists(cached):
        from data.utils import ffmpeg_once
        os.makedirs(os.path.dirname(cached), exist_ok=True)
        ffmpeg_once(video, cached, fps=fps, resolution=res)
    live.load_video(cached)
    embed = model.get_input_embeddings()
    interval = embed(torch.tensor([[live.frame_token_interval_id]], device='cuda')) # ',' token that follows each frame in streaming

    with torch.no_grad():
        # vision features for every frame once, so the cache fill and the timed steps only run the LLM
        frames = torch.cat([model.visual_embed(live.video_tensor[i:i + 64]).view(-1, live.frame_num_tokens, live.hidden_size)
                            for i in range(0, live.num_video_frames, 64)])
        del live.video_tensor
        print(f'device: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | {len(frames)} frames of features')
        print(f'SDPA backends enabled: flash={torch.backends.cuda.flash_sdp_enabled()} mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()} '
              f'cudnn={torch.backends.cuda.cudnn_sdp_enabled()} math={torch.backends.cuda.math_sdp_enabled()}')

        def frame_step(pkv, i):
            x = torch.cat([interval, frames[i % len(frames)].unsqueeze(0)], dim=1) # 1 + 10 tokens
            return model(inputs_embeds=x, use_cache=True, past_key_values=pkv).past_key_values

        def time_steps(pkv, i0, n=STEPS):
            for i in range(3): # warm-up
                pkv = frame_step(pkv, i0 + i)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for i in range(n):
                pkv = frame_step(pkv, i0 + 3 + i)
            torch.cuda.synchronize()
            return pkv, 1000 * (time.perf_counter() - t0) / n

        pkv = model(inputs_embeds=embed(live._start_ids), use_cache=True).past_key_values
        nxt, results = 0, []
        print('\n%10s %14s %14s' % ('kv_tokens', 'prefill_ms', 'allocated_GB'))
        for target in KV_TARGETS:
            while kv_tokens(pkv) + CHUNK_FRAMES * 11 <= target:
                chunk = torch.cat([torch.cat([interval[0], frames[(nxt + j) % len(frames)]]) for j in range(CHUNK_FRAMES)]).unsqueeze(0)
                pkv = model(inputs_embeds=chunk, use_cache=True, past_key_values=pkv).past_key_values
                nxt += CHUNK_FRAMES
            pkv, ms = time_steps(pkv, nxt)
            results.append({'kv_tokens': kv_tokens(pkv), 'prefill_ms': ms, 'allocated_gb': torch.cuda.memory_allocated() / 1e9})
            print('%10d %14.1f %14.1f' % (results[-1]['kv_tokens'], ms, results[-1]['allocated_gb']))

        # 1. attention kernels, at the longest cache
        from torch.nn.attention import sdpa_kernel, SDPBackend
        print(f'\nSDPA backend timing at {kv_tokens(pkv)} tokens:')
        backends = {}
        for backend in [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]:
            try:
                with sdpa_kernel(backend):
                    pkv, ms = time_steps(pkv, nxt, n=10)
                backends[backend.name] = ms
                print(f'  {backend.name:22s} {ms:8.1f} ms/frame')
            except Exception as e:
                backends[backend.name] = None
                print(f'  {backend.name:22s} not usable here: {str(e).splitlines()[0][:100]}')

        # 2. where the time goes
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for i in range(10):
                pkv = frame_step(pkv, nxt + i)
            torch.cuda.synchronize()
        trace = os.path.join(OUT, 'prefill_trace.json')
        prof.export_chrome_trace(trace)
        print(f'\nTop ops by GPU time, 10 frame prefills at {kv_tokens(pkv)} tokens:')
        print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=20, max_name_column_width=60))
        print(f'Top ops by CPU time:')
        print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=15, max_name_column_width=60))

        # 3. LoRA merged into the base weights (the cache stays valid: same model, fewer matmuls)
        merged_ms = None
        if hasattr(model, 'merge_and_unload'):
            model = model.merge_and_unload()
            pkv, merged_ms = time_steps(pkv, nxt)
            print(f'\nLoRA merged: {merged_ms:.1f} ms/frame at {kv_tokens(pkv)} tokens (unmerged: {results[-1]["prefill_ms"]:.1f} ms at {results[-1]["kv_tokens"]})')

    summary = {'device': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'growth': results, 'sdpa_backends_ms': backends,
               'merged_lora_ms': merged_ms, 'peak_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
               'peak_reserved_gb': torch.cuda.max_memory_reserved() / 1e9}
    json.dump(summary, open(os.path.join(OUT, 'prefill_summary.json'), 'w'), indent=1)
    print(f"\npeak memory: {summary['peak_allocated_gb']:.1f} GB allocated, {summary['peak_reserved_gb']:.1f} GB reserved")
    print(f'Saved {OUT}/prefill_summary.json and {trace} (open in chrome://tracing or https://ui.perfetto.dev)')

if __name__ == '__main__':
    main()
