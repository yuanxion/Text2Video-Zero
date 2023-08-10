import os
import shlex
import sys
import torch
from pathlib import Path


def read_and_splite(story_file):
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

        print(f'[line {i}] {line}')
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

        # if i == 4:
        #     sys.exit(0)

    for j, sentence in enumerate(sentences):
        print(f'  [sentence {j}] {sentence}')

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


story = 'demo/tadpoles_look_for_mom.txt'
sentences = read_and_splite(story)
#demo_t2v(sentence)
