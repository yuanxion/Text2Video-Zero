import os
import sys
import torch
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
os.environ['CURL_CA_BUNDLE'] = ''

from model import Model

model = Model(device = "cuda", dtype = torch.float16)
#model = Model(device = "cpu", dtype = torch.float32)
print(f'--> model {model}')

prompt = "A horse galloping on a street"
params = {"t0": 44, "t1": 47 , "motion_field_strength_x" : 12, "motion_field_strength_y" : 12, "video_length": 8}
# more options
params.update({"chunk_size" : 1})
params.update({"merging_ratio" : 1})
print(f'params: {params}')

#out_path, fps = f"./text2video_{prompt.replace(' ','_')}.mp4", 4
#model.process_text2video(prompt, fps = fps, path = out_path, **params)

motion_path = '__assets__/poses_skeleton_gifs/dance1_corr.mp4'
out_path = f"./text2video_pose_guidance_{prompt.replace(' ','_')}.gif"
model.process_controlnet_pose(motion_path, prompt=prompt, save_path=out_path)
