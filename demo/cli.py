import os, torchvision, transformers, tqdm, time, json
import torch.multiprocessing as mp

from data.utils import ffmpeg_once

from .inference import LiveInfer, system_available_gb
logger = transformers.logging.get_logger('liveinfer')

# python -m demo.cli --resume_from_checkpoint ... 

def main(liveinfer: LiveInfer):
    src_video_path = os.environ.get('VIDEO', 'demo/assets/cooking.mp4') # VIDEO=<path> to profile another (e.g. longer) video
    name, ext = os.path.splitext(src_video_path)
    ffmpeg_video_path = os.path.join('demo/assets/cache', name + f'_{liveinfer.frame_fps:g}fps_{liveinfer.frame_resolution}' + ext)
    # OUT_PREFIX=<path without extension>: where the history / phase / growth files go (default: next to the video)
    save_history_path = os.environ.get('OUT_PREFIX', name) + '.json'
    if not os.path.exists(ffmpeg_video_path):
        os.makedirs(os.path.dirname(ffmpeg_video_path), exist_ok=True)
        ffmpeg_once(src_video_path, ffmpeg_video_path, fps=liveinfer.frame_fps, resolution=liveinfer.frame_resolution)
        logger.warning(f'{src_video_path} -> {ffmpeg_video_path}, {liveinfer.frame_fps} FPS, {liveinfer.frame_resolution} Resolution')
    
    liveinfer.load_video(ffmpeg_video_path)
    if os.environ.get('WARMUP', '1') != '0': # WARMUP=0: keep the cold start inside the measured session
        liveinfer.warmup()
    liveinfer.input_query_stream('Please narrate the video in real time.', video_time=0.0)
    # QUERY_EVERY=<seconds>: also ask a question at fixed video times, so decode cost is sampled evenly as the KV cache grows
    query_every = float(os.environ.get('QUERY_EVERY', 0))
    if query_every > 0:
        for t in range(1, int(liveinfer.video_duration // query_every) + 1):
            liveinfer.input_query_stream('What is the person doing now?', video_time=t * query_every)
    # liveinfer.input_query_stream('Hi, who are you?', video_time=1.0)
    # liveinfer.input_query_stream('Yes, I want to check its safety.', video_time=3.0)
    # liveinfer.input_query_stream('No, I am going to install something to alert pedestrians to move aside. Could you guess what it is?', video_time=12.5)

    # MAX_MINUTES=<n>: stop after n minutes of video. MIN_FREE_GB=<g> (default 8): stop cleanly when the system's available
    # memory drops below g GB; on unified-memory devices running out can hang the whole machine.
    num_frames = liveinfer.num_video_frames
    if os.environ.get('MAX_MINUTES'):
        num_frames = min(num_frames, int(float(os.environ['MAX_MINUTES']) * 60 * liveinfer.frame_fps))
    min_free_gb = float(os.environ.get('MIN_FREE_GB', 8))
    growth_path = save_history_path.replace('.json', '_growth.json')
    timecosts = []
    pbar = tqdm.tqdm(total=num_frames, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}]")
    history = {'video_path': src_video_path, 'frame_fps': liveinfer.frame_fps, 'conversation': []}
    # wall-clock and perf_counter at the start and end of streaming, to line phase events up with an external power log
    session = {'wall_start': time.time(), 'perf_start': time.perf_counter()}
    for i in range(num_frames):
        if liveinfer.phase_timing and i and i % int(60 * liveinfer.frame_fps) == 0:
            json.dump(liveinfer.growth_events, open(growth_path, 'w'), indent=1)  # every minute of video, so a crash keeps the data
            free = system_available_gb()
            if free is not None and free < min_free_gb:
                print(f'\n!!! only {free:.1f} GB of system memory available (< MIN_FREE_GB={min_free_gb}); stopping at {i / liveinfer.frame_fps / 60:.1f} min of video')
                break
    # for i in range(100):
        # liveinfer.frame_token_interval_threshold -= 0.00175 # decay
        start_time = time.time()
        liveinfer.input_video_stream(i / liveinfer.frame_fps)
        query, response = liveinfer()
        end_time = time.time()
        timecosts.append(end_time - start_time)
        fps = (i + 1) / sum(timecosts)
        pbar.set_postfix_str(f"Average Processing FPS: {fps:.1f}")
        pbar.update(1)
        if query:
            history['conversation'].append({'role': 'user', 'content': query, 'time': liveinfer.video_time, 'fps': fps, 'cost': timecosts[-1]})
            print(query)
        if response:
            history['conversation'].append({'role': 'assistant', 'content': response, 'time': liveinfer.video_time, 'fps': fps, 'cost': timecosts[-1]})
            print(response)
        if not query and not response:
            history['conversation'].append({'time': liveinfer.video_time, 'fps': fps, 'cost': timecosts[-1]})
    session.update(wall_end=time.time(), perf_end=time.perf_counter(), frames=len(timecosts), video_seconds=len(timecosts) / liveinfer.frame_fps)
    history['session'] = session
    json.dump(history, open(save_history_path, 'w'), indent=4)
    print(f'The conversation history has been saved to {save_history_path}.')

    if liveinfer.phase_timing:
        import torch
        from profiling_scripts.timeline import print_phase_summary, plot_phase_timeline
        print(f'peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB allocated, {torch.cuda.max_memory_reserved() / 1e9:.1f} GB reserved')
        print_phase_summary(liveinfer.phase_events)
        events_path = save_history_path.replace('.json', '_phase_events.json')
        json.dump(liveinfer.phase_events, open(events_path, 'w'), indent=2)
        print(f'Raw phase events saved to {events_path}.')
        growth_path = save_history_path.replace('.json', '_growth.json')
        json.dump(liveinfer.growth_events, open(growth_path, 'w'), indent=1)
        print_growth_summary(liveinfer)
        print(f'Per-frame KV cache / memory growth saved to {growth_path}.')
        try:
            plot_path = save_history_path.replace('.json', '_phase_timeline.png')
            plot_phase_timeline(liveinfer.phase_events, plot_path)
        except ImportError:
            print('matplotlib not installed: skipped the timeline plot')

def print_growth_summary(liveinfer: LiveInfer):
    # how frame prefill and decode cost grow with the KV cache, one row per minute of video
    config = liveinfer.model.config
    head_dim = getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads
    kv_bytes_per_token = 2 * config.num_hidden_layers * config.num_key_value_heads * head_dim * 2 # K and V, bf16
    print(f'\nKV cache: {kv_bytes_per_token / 1024:.0f} KB per token, LLM context trained up to {config.max_position_embeddings} tokens')
    print('%8s %10s %8s %13s %12s %13s %15s %18s' % ('minute', 'kv_tokens', 'kv_GB', 'allocated_GB', 'reserved_GB', 'sys_avail_GB', 'prefill_ms/frm', 'decode_ms/token'))
    rows = {}
    for e in liveinfer.growth_events:
        rows.setdefault(int(e['video_time'] // 60), []).append(e)
    for minute, events in sorted(rows.items()):
        prefill = [e['seconds'] for e in events if e['phase'] == 'frame_prefill']
        decode = [(e['seconds'], e['units']) for e in events if e['phase'] == 'decode']
        kv = max(e['kv_tokens'] for e in events)
        decode_ms = '%.1f' % (1000 * sum(s for s, _ in decode) / sum(u for _, u in decode)) if decode else '-'
        avail = [e['system_available_gb'] for e in events if e.get('system_available_gb') is not None]
        print('%8d %10d %8.2f %13.1f %12.1f %13s %15.1f %18s' % (minute, kv, kv * kv_bytes_per_token / 1e9, max(e['allocated_gb'] for e in events),
                                                   max(e.get('reserved_gb', 0) for e in events), '%.1f' % min(avail) if avail else '-',
                                                   1000 * sum(prefill) / max(len(prefill), 1), decode_ms))

if __name__ == '__main__':
    liveinfer = LiveInfer()
    main(liveinfer)