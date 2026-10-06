import os, torchvision, transformers, tqdm, time, json
import torch.multiprocessing as mp

from data.utils import ffmpeg_once

from .inference import LiveInfer
logger = transformers.logging.get_logger('liveinfer')

# python -m demo.cli --resume_from_checkpoint ... 

def main(liveinfer: LiveInfer):
    src_video_path = liveinfer.demo_video_path
    name, ext = os.path.splitext(src_video_path)
    ffmpeg_video_path = os.path.join('demo/assets/cache', name + f'_{liveinfer.frame_fps}fps_{liveinfer.frame_resolution}' + ext)
    save_history_path = src_video_path.replace('.mp4', '.json')

    power_meter = None
    if liveinfer.phase_timing:
        from profiling_scripts.power_meter import PowerMeter
        power_meter = PowerMeter()
        power_meter.__enter__()

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
        power_meter.__exit__(None, None, None)

        from profiling_scripts.timeline import (
            print_phase_summary, plot_phase_timeline,
            print_phase_energy_summary, plot_phase_timeline_with_power,
            plot_phase_timeline_overlay,
        )
        print_phase_summary(liveinfer.phase_events)
        print_phase_energy_summary(liveinfer.phase_events, power_meter)

        video_title = os.path.basename(name)
        plot_path = save_history_path.replace('.json', '_phase_timeline.pdf')
        plot_phase_timeline(liveinfer.phase_events, plot_path, title=video_title)
        power_plot_path = save_history_path.replace('.json', '_phase_power_timeline.pdf')
        plot_phase_timeline_with_power(liveinfer.phase_events, power_meter.samples, power_plot_path, title=video_title)
        overlay_plot_path = save_history_path.replace('.json', '_phase_power_overlay.pdf')
        plot_phase_timeline_overlay(liveinfer.phase_events, power_meter.samples, overlay_plot_path, title=video_title)

        events_path = save_history_path.replace('.json', '_phase_events.json')
        json.dump(liveinfer.phase_events, open(events_path, 'w'), indent=2)
        print(f'Raw phase events saved to {events_path}.')

        power_events_path = save_history_path.replace('.json', '_power_samples.json')
        json.dump(power_meter.samples, open(power_events_path, 'w'), indent=2)
        print(f'Raw power samples saved to {power_events_path}.')

if __name__ == '__main__':
    liveinfer = LiveInfer()
    main(liveinfer)