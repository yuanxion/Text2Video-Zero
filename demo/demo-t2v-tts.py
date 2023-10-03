import librosa
import math
import os
import re
import shlex
import subprocess
import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).absolute().parent.parent))
from model import Model

txt_folder = './texts'
mp3_folder = './voices'
mp4_folder = './videos'


def convert_string(prompt: str) -> str:
    return prompt.replace(', ', '_').replace(' ', '_')


def clear_folders(folders) -> None:
    for folder in folders:
        if not os.path.exists(folder):
            print(f'{folder=} is not existed, continue...')
            continue
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)


def run_command(command):
    print(f'[*] run_command: {" ".join(command)} ')

    try:
        result = subprocess.run(command, shell=False, timeout=6000)
        print(f'[v] subprocess {result = }')
    except subprocess.CalledProcessError as e:
        print('[x] Command: {" ".join(command)}')
        print('[x] Exception return code: {e.returncode}')
        print('[x] Exception output: {e.output}')


def read_and_splite(story_file, txt_folder: Path):
    print(f'[+] read_and_splite')

    with open(story_file, 'r') as f:
        lines = f.readlines()
        if lines is None:
            print(f'Read story file failed, exit...')
            sys.exit(0)

    shots, voiceovers = [], []
    ps = re.compile(r'[[](.*?)[]]', re.S)
    pv = re.compile(r'["](.*?)["]', re.S)
    for i, line in enumerate(lines):
        #print(f'--> {line = }')

        shot = re.findall(ps, line)[0]
        shot = shot.strip('.')
        shots.append(shot)

        voiceover = re.findall(pv, line)[0]
        voiceover = voiceover.strip('.')
        voiceovers.append(voiceover)

    return shots, voiceovers


def demo_t2v(prompt: str, duration: int, mp4_folder: Path):
    # prompt = "春天来了"
    print(f'[-] demo_t2v {prompt=}')

    mp4_file = f'{mp4_folder}/{convert_string(prompt)}.mp4'
    curr_dir = os.getcwd()
    voice_dir = os.path.join(curr_dir, mp4_folder)
    os.makedirs(voice_dir, exist_ok=True)

    print(f'\n###### demo_t2v', sys._getframe().f_code.co_name)
    model = Model(device="cuda", dtype=torch.float16)

    out_path, fps = mp4_file, 1 #4
    video_length = math.ceil(duration * 30 / fps)
    print(f'{duration = } {video_length = }')
    params = {
        "t0": 40,
        "t1": 45,
        "motion_field_strength_x": 6,
        "motion_field_strength_y": 6,
        "video_length": video_length,
        #"video_length": 8,
    }
    # more options for low GPU memory usage
    # params.update({"chunk_size": 2})  # 8
    params.update({"chunk_size": 4})  # 8
    params.update({"merging_ratio": 0.4})  # 0
    params.update({"num_inference_steps": 20})  # 50

    model.process_text2video(prompt, fps=fps, path=out_path, **params)

    return mp4_file


def demo_tts(voiceover: str, mp3_folder: Path):
    print(f'[-] demo_tts {voiceover=}')

    mp3_file = f'{mp3_folder}/{convert_string(voiceover)}.mp3'
    tts_file = f'{mp3_folder}/{convert_string(voiceover)}.tts'
    curr_dir = os.getcwd()
    voice_dir = os.path.join(curr_dir, mp3_folder)
    os.makedirs(voice_dir, exist_ok=True)

    command = ['edge-tts']
    command += ['--voice', 'zh-CN-XiaoyiNeural']
    command += ['--text', voiceover]
    command += ['--write-media', mp3_file]
    command += ['--write-subtitles', tts_file]
    run_command(command)

    return mp3_file, tts_file


def merge_video_audio(
    shot: str,
    mp3_file: Path,
    tts_file: Path,
    mp4_file: Path,
    mp4_folder: Path,
    output_file: Path,
):
    print(f'[-] merge_video_audio')

    merged_file = f'{mp4_folder}/merged_{convert_string(shot)}.mp4'

    # merge video and audio
    command = ['ffmpeg']
    command += ['-i', mp4_file]
    command += ['-i', mp3_file]
    command += ['-c:v', 'copy']
    command += ['-c:a', 'mp3']
    command += [merged_file]
    run_command(command)

    # add subtitles
    command = ['ffmpeg']
    command += ['-i', merged_file]
    # command += [
    #    '-vf',
    #    f'drawtext=text={shot} :x=50:y=400:fontsize=24:fontcolor=white',
    # ]
    command += ['-filter_complex', f"subtitles={tts_file}"]
    command += ['-max_muxing_queue_size', '1024']
    command += [f'{mp4_folder}/{output_file}']
    run_command(command)


def merge_story_videos(
    story_file: Path, video_list_file: Path, mp4_folder: Path
) -> None:
    print(f'[+] merge_story_videos for {story_file}')

    output_file = f'{mp4_folder}/{story_file.split("/")[-1].split(".")[0]}.mp4'

    command = ['ffmpeg']
    command += ['-f', 'concat']
    command += ['-safe', '0']
    command += [f'-i', video_list_file]
    command += ['-c:v', 'copy']
    command += ['-c:a', 'copy']
    command += ['-max_muxing_queue_size', '1024']
    command += [output_file]
    run_command(command)


def demo_t2v_tts(shot: str, voiceover: str, mp3_folder: Path, mp4_folder: Path):
    print(f'[+] demo_t2v_tts {shot=}')
    output_file = f'final_{convert_string(shot)}.mp4'

    mp3_file, tts_file = demo_tts(voiceover, mp3_folder)
    duration = librosa.get_duration(path=mp3_file)
    print(f'{mp3_file = }: {duration = }')
    mp4_file = demo_t2v(shot, math.ceil(duration), mp4_folder)
    merge_video_audio(
        shot, mp3_file, tts_file, mp4_file, mp4_folder, output_file
    )

    return output_file


if __name__ == '__main__':
    # story_file = 'demo/tadpoles_look_for_mom.txt'
    #story_file = 'demo/protect_yourself_from_covid19.txt'
    #story_file = 'demo/tell_me_a_story.txt'
    story_file = 'demo/chatgpt4_story.txt'

    clear_folders([txt_folder, mp3_folder, mp4_folder])
    shots, voiceovers = read_and_splite(story_file, txt_folder)

    video_list_file = f'{mp4_folder}/list.txt'
    for shot, voiceover in zip(shots, voiceovers):
        print(f'{shot = } {voiceover = }')

        # start = input('Start to run? (y/n) ')
        # if start == 'n':
        #     sys.exit(0)
        # elif start == '':
        #     continue

        video_merged = demo_t2v_tts(shot, voiceover, mp3_folder, mp4_folder)
        line = f'file "{video_merged}"'

        command = f'echo {line} >> {video_list_file}'
        print(f'{command=}')
        os.system(f"{command}")

    # merge all small videos (with voice and subtitles)
    merge_story_videos(story_file, video_list_file, mp4_folder)


# python demo/demo-t2v-tts.py
