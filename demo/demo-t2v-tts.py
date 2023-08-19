import os
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
        print(f'[v] demo_tts subprocess {result=}')
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

    sentences = []
    for i, line in enumerate(lines):
        line = line.strip('\n')

        # skip blank line
        if line == '':
            continue

        # print(f'[line {i}] {line}')
        lex = shlex.shlex(line, posix=True)
        lex.whitespace = '.'
        lex.whitespace_split = True
        lex.quotes = '"'
        res = list(lex)

        for sentence in res:
            # skip blank sentence
            if sentence == '':
                continue

            # remove white-space
            sentence = sentence.strip()
            sentences.append(sentence)

    # for j, sentence in enumerate(sentences):
    #     print(f'  [sentence {j}] {sentence}')

    return sentences


def demo_t2v(prompt: str, mp4_folder: Path):
    # prompt = "春天来了"
    print(f'[-] demo_t2v {prompt=}')

    mp4_file = f'{mp4_folder}/{convert_string(prompt)}.mp4'
    curr_dir = os.getcwd()
    voice_dir = os.path.join(curr_dir, mp4_folder)
    os.makedirs(voice_dir, exist_ok=True)

    print(f'\n###### demo_t2v', sys._getframe().f_code.co_name)

    model = Model(device="cuda", dtype=torch.float16)

    params = {
        "t0": 44,
        "t1": 47,
        "motion_field_strength_x": 12,
        "motion_field_strength_y": 12,
        "video_length": 8,
        # "video_length": 80,
    }
    # more options for low GPU memory usage
    # params.update({"chunk_size": 2})  # 8
    params.update({"chunk_size": 4})  # 8
    params.update({"merging_ratio": 1})  # 0

    out_path, fps = mp4_file, 4
    model.process_text2video(prompt, fps=fps, path=out_path, **params)

    return mp4_file


def demo_tts(prompt: str, mp3_folder: Path):
    print(f'[-] demo_tts {prompt=}')

    mp3_file = f'{mp3_folder}/{convert_string(prompt)}.mp3'
    tts_file = f'{mp3_folder}/{convert_string(prompt)}.tts'
    curr_dir = os.getcwd()
    voice_dir = os.path.join(curr_dir, mp3_folder)
    os.makedirs(voice_dir, exist_ok=True)

    command = ['edge-tts']
    command += ['--voice', 'zh-CN-XiaoyiNeural']
    command += ['--text', prompt]
    command += ['--write-media', mp3_file]
    command += ['--write-subtitles', tts_file]
    run_command(command)

    return mp3_file, tts_file


def merge_video_audio(
    prompt: str,
    mp3_file: Path,
    tts_file: Path,
    mp4_file: Path,
    mp4_folder: Path,
    output_file: Path,
):
    print(f'[-] merge_video_audio')

    merged_file = f'{mp4_folder}/merged_{convert_string(prompt)}.mp4'

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
    #    f'drawtext=text={prompt} :x=50:y=400:fontsize=24:fontcolor=white',
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


def demo_t2v_tts(sentence, mp3_folder: Path, mp4_folder: Path):
    print(f'[+] demo_t2v_tts {sentence=}')
    output_file = f'final_{convert_string(sentence)}.mp4'

    mp3_file, tts_file = demo_tts(sentence, mp3_folder)
    mp4_file = demo_t2v(sentence, mp4_folder)
    merge_video_audio(
        sentence, mp3_file, tts_file, mp4_file, mp4_folder, output_file
    )

    return output_file


if __name__ == '__main__':
    # story_file = 'demo/tadpoles_look_for_mom.txt'
    story_file = 'demo/protect_yourself_from_covid19.txt'

    clear_folders([txt_folder, mp3_folder, mp4_folder])

    sentences = read_and_splite(story_file, txt_folder)

    video_list_file = f'{mp4_folder}/list.txt'
    for sentence in sentences:
        print(f'{sentence = }')
        # start = input('Start to run? (y/n) ')
        # if start == 'n':
        #     sys.exit(0)
        # elif start == '':
        #     continue

        video_merged = demo_t2v_tts(sentence, mp3_folder, mp4_folder)
        line = f'file "{video_merged}"'

        command = f'echo {line} >> {video_list_file}'
        print(f'{command=}')
        os.system(f"{command}")

    # merge all small videos (with voice and subtitles)
    merge_story_videos(story_file, video_list_file, mp4_folder)


# python demo/demo-t2v-tts.py
