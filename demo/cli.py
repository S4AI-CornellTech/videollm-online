import os, torchvision, transformers, tqdm, time, json
import torch.multiprocessing as mp

from data.utils import ffmpeg_once

from .inference import LiveInfer
logger = transformers.logging.get_logger('liveinfer')

# python -m demo.cli --resume_from_checkpoint ... 

def main(liveinfer: LiveInfer):
    src_video_path = os.environ.get('VIDEO', 'demo/assets/cooking.mp4') # VIDEO=<path> to profile another (e.g. longer) video
    name, ext = os.path.splitext(src_video_path)
    ffmpeg_video_path = os.path.join('demo/assets/cache', name + f'_{liveinfer.frame_fps}fps_{liveinfer.frame_resolution}' + ext)
    save_history_path = src_video_path.replace('.mp4', '.json')
    if not os.path.exists(ffmpeg_video_path):
        os.makedirs(os.path.dirname(ffmpeg_video_path), exist_ok=True)
        ffmpeg_once(src_video_path, ffmpeg_video_path, fps=liveinfer.frame_fps, resolution=liveinfer.frame_resolution)
        logger.warning(f'{src_video_path} -> {ffmpeg_video_path}, {liveinfer.frame_fps} FPS, {liveinfer.frame_resolution} Resolution')
    
    liveinfer.load_video(ffmpeg_video_path)
    liveinfer.input_query_stream('Please narrate the video in real time.', video_time=0.0)
    # liveinfer.input_query_stream('Hi, who are you?', video_time=1.0)
    # liveinfer.input_query_stream('Yes, I want to check its safety.', video_time=3.0)
    # liveinfer.input_query_stream('No, I am going to install something to alert pedestrians to move aside. Could you guess what it is?', video_time=12.5)

    timecosts = []
    pbar = tqdm.tqdm(total=liveinfer.num_video_frames, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}]")
    history = {'video_path': src_video_path, 'frame_fps': liveinfer.frame_fps, 'conversation': []} 
    for i in range(liveinfer.num_video_frames):
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
    print('%8s %10s %8s %13s %15s %18s' % ('minute', 'kv_tokens', 'kv_GB', 'allocated_GB', 'prefill_ms/frm', 'decode_ms/token'))
    rows = {}
    for e in liveinfer.growth_events:
        rows.setdefault(int(e['video_time'] // 60), []).append(e)
    for minute, events in sorted(rows.items()):
        prefill = [e['seconds'] for e in events if e['phase'] == 'frame_prefill']
        decode = [(e['seconds'], e['units']) for e in events if e['phase'] == 'decode']
        kv = max(e['kv_tokens'] for e in events)
        decode_ms = '%.1f' % (1000 * sum(s for s, _ in decode) / sum(u for _, u in decode)) if decode else '-'
        print('%8d %10d %8.2f %13.1f %15.1f %18s' % (minute, kv, kv * kv_bytes_per_token / 1e9, max(e['allocated_gb'] for e in events),
                                                   1000 * sum(prefill) / max(len(prefill), 1), decode_ms))

if __name__ == '__main__':
    liveinfer = LiveInfer()
    main(liveinfer)