import os
import shlex
import subprocess
import sys
import torch
from pathlib import Path


def read_and_splite(story_file, txt_folder: Path):
    with open(story_file, 'r') as f:
        lines = f.readlines()

    sentences = []
    for i, line in enumerate(lines):
        line = line.strip('\n')

        # skip blank line
        if line == '':
            continue
        #if not i == 12:
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

def demo_t2v(prompt: str):
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

    out_path, fps = f"./text2video_t2v_tts_{prompt.replace(' ','_')}.mp4", 4
    model.process_text2video(prompt, fps=fps, path=out_path, **params)


def demo_tts(prompt: str, mp3_folder: Path):
    mp3_file = f'{mp3_folder}/{prompt}.mp3'
    tts_file = f'{mp3_folder}/{prompt}.tts'

    curr_dir = os.getcwd()
    voice_dir = os.path.join(curr_dir, mp3_folder)
    os.makedirs(voice_dir, exist_ok=True)

    command = ['edge-tts']
    command += ['--voice', 'zh-CN-XiaoyiNeural']
    command += ['--text', prompt]
    command += ['--write-media', mp3_file]
    command += ['--write-subtitles', tts_file]

    try:
        result = subprocess.run(command, timeout=10)
        print(result)
    except subprocess.CalledProcessError as e:
        print('Command: {command}')
        print('Exception return code: {e.returncode}')
        print('Exception output: {e.output}')

def demo_t2v_tts(sentence, mp3_folder: Path, mp4_folder: Path):
    print(f'demo_t2v_tts {sentence=}')
    demo_tts(sentence, mp3_folder)

txt_folder = './texts'
mp3_folder = './voices'
mp4_folder = './videos'

story = 'demo/tadpoles_look_for_mom.txt'
sentences = read_and_splite(story, txt_folder)

for sentence in sentences:
    demo_t2v_tts(sentence, mp3_folder, mp4_folder)
