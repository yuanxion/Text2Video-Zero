import os
import shlex
import subprocess
import sys
import torch
from pathlib import Path


def run_command(command):
    print(f'[*] run_command {command=} ')

    try:
        result = subprocess.run(command, shell=False, timeout=10)
        print(f'demo_tts subprocess {result=}')
    except subprocess.CalledProcessError as e:
        print('Command: {command}')
        print('Exception return code: {e.returncode}')
        print('Exception output: {e.output}')


def read_and_splite(story_file, txt_folder: Path):
    print(f'[+] read_and_splite')

    with open(story_file, 'r') as f:
        lines = f.readlines()

    sentences = []
    for i, line in enumerate(lines):
        line = line.strip('\n')

        # skip blank line
        if line == '':
            continue
        # if not i == 12:
        #    continue

        # print(f'[line {i}] {line}')
        lex = shlex.shlex(line, posix=True)
        lex.whitespace = '.'
        lex.whitespace_split = True
        lex.quotes = '"'
        res = list(lex)
        # # sq = shlex.quote(line)
        # # res = shlex.split(sq, comments='.', posix=True)
        # print(f'res {i}: {res=}')

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
    print(f'[-] demo_t2v {prompt=}')

    mp4_file = f'{mp4_folder}/{prompt.replace(" ", "_")}.mp4'
    curr_dir = os.getcwd()
    voice_dir = os.path.join(curr_dir, mp4_folder)
    os.makedirs(voice_dir, exist_ok=True)

    sys.path.append(str(Path(__file__).absolute().parent.parent))
    from model import Model

    print(f'\n######', sys._getframe().f_code.co_name)

    model = Model(device="cuda", dtype=torch.float16)

    # prompt = "春天来了"
    params = {
        "t0": 44,
        "t1": 47,
        "motion_field_strength_x": 12,
        "motion_field_strength_y": 12,
        "video_length": 8,
        # "video_length": 80,
    }
    # more options for low GPU memory usage
    params.update({"chunk_size": 2})  # 8
    params.update({"merging_ratio": 1})  # 0

    out_path, fps = mp4_file, 4
    model.process_text2video(prompt, fps=fps, path=out_path, **params)

    return mp4_file


def demo_tts(prompt: str, mp3_folder: Path):
    print(f'[-] demo_tts {prompt=}')

    mp3_file = f'{mp3_folder}/{prompt.replace(" ", "_")}.mp3'
    tts_file = f'{mp3_folder}/{prompt.replace(" ", "_")}.tts'
    curr_dir = os.getcwd()
    voice_dir = os.path.join(curr_dir, mp3_folder)
    os.makedirs(voice_dir, exist_ok=True)

    command = ['edge-tts']
    command += ['--voice', 'zh-CN-XiaoyiNeural']
    command += ['--text', prompt]
    command += ['--write-media', mp3_file]
    command += ['--write-subtitles', tts_file]

    run_command(command)
    return mp3_file


def merge_video_audio(mp3_file: Path, mp4_file: Path, output_file: Path):
    print(f'[-] merge_video_audio')

    # command = ['ffmpeg -i videos/Spring_has_come.mp4 -i voices/Spring_has_come.mp3 -c:v copy -c:a mp3 merged.mp4']

    command = ['ffmpeg']
    command += ['-i', mp4_file]
    command += ['-i', mp3_file]
    command += ['-c:v', 'copy']
    command += ['-c:a', 'mp3']
    command += [output_file]

    run_command(command)


def demo_t2v_tts(sentence, mp3_folder: Path, mp4_folder: Path):
    print(f'[+] demo_t2v_tts {sentence=}')
    output_file = f'merged.mp4'

    # mp3_file = demo_tts(sentence, mp3_folder)
    # mp4_file = demo_t2v(sentence, mp4_folder)
    # merge_video_audio(mp3_file, mp4_file, output_file)

    merge_video_audio(
        './voices/Spring_has_come.mp3',
        './videos/Spring_has_come.mp4',
        output_file,
    )


txt_folder = './texts'
mp3_folder = './voices'
mp4_folder = './videos'

story = 'demo/tadpoles_look_for_mom.txt'
sentences = read_and_splite(story, txt_folder)

for sentence in sentences:
    demo_t2v_tts(sentence, mp3_folder, mp4_folder)

    # for testing
    sys.exit(0)

# python demo/demo-t2v-tts.py
