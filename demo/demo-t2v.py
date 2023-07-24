import os
import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).absolute().parent.parent))
# os.environ['CURL_CA_BUNDLE'] = ''
from model import Model


def prepare_model_list():
    from hf_utils import get_model_list

    model_list = get_model_list()
    for idx, name in enumerate(model_list):
        print(idx, name)
    idx = int(
        input("Select the model by the listed number: ")
    )  # select the model of your choice


def demo_t2v():
    print(f'\n######', sys._getframe().f_code.co_name)
    model = Model(device="cuda", dtype=torch.float16)

    prompt = "A horse galloping on a raining street"
    params = {
        "t0": 44,
        "t1": 47,
        "motion_field_strength_x": 12,
        "motion_field_strength_y": 12,
        "video_length": 8,
    }
    # more options for low GPU memory usage
    params.update({"chunk_size": 2})  # 8
    params.update({"merging_ratio": 1})  # 0

    out_path, fps = f"./text2video_{prompt.replace(' ','_')}.mp4", 4
    model.process_text2video(prompt, fps=fps, path=out_path, **params)


# Pose
def demo_t2v_with_pose():
    print(f'\n######', sys._getframe().f_code.co_name)
    model = Model(device="cuda", dtype=torch.float16)
    params = {}
    # more options for low GPU memory usage
    params.update({"chunk_size": 2})  # 8
    params.update({"merging_ratio": 1})  # 0
    params.update({"resolution": 460})  # 512

    prompt = 'an astronaut dancing in outer space'
    motion_path = '__assets__/poses_skeleton_gifs/dance1_corr.mp4'
    # out_path = f"./text2video_pose_guidance_{prompt.replace(' ','_')}.gif"
    out_path = f"./text2video_pose_guidance_{prompt.replace(' ','_')}.mp4"
    model.process_controlnet_pose(
        motion_path, prompt=prompt, save_path=out_path, **params
    )


# Edge
def demo_t2v_with_edge():
    print(f'\n######', sys._getframe().f_code.co_name)
    model = Model(device="cuda", dtype=torch.float16)
    params = {"low_threshold": 100, "high_threshold": 200}
    # more options for low GPU memory usage
    params.update({"chunk_size": 4})  # 8
    params.update({"merging_ratio": 0})  # 0
    params.update({"resolution": 400})  # 512

    prompt = 'oil painting of a deer, a high-quality, detailed, and professional photo'
    # video_path = '__assets__/canny_videos_mp4/deer.mp4'
    video_path = '__assets__/canny_videos_edge/deer.mp4'
    out_path = f'./text2video_edge_guidance_{prompt}.mp4'
    model.process_controlnet_canny(
        video_path, prompt=prompt, save_path=out_path, **params
    )


# Edge, DreamBooth
def demo_t2v_with_edge_dreambooth():
    print(f'\n######', sys._getframe().f_code.co_name)
    model = Model(device="cuda", dtype=torch.float16)
    params = {"low_threshold": 100, "high_threshold": 200}
    # more options for low GPU memory usage
    params.update({"chunk_size": 4})  # 8
    params.update({"merging_ratio": 0})  # 0
    params.update({"resolution": 400})  # 512

    prompt = 'avatar style'
    video_path = 'woman1'
    dreambooth_model_path = 'Avatar DB'
    out_path = f'./text2video_edge_db_{prompt}.gif'

    model.process_controlnet_canny_db(
        dreambooth_model_path,
        video_path,
        prompt=prompt,
        save_path=out_path,
        **params,
    )


# Depth
def demo_t2v_with_depth():
    print(f'\n######', sys._getframe().f_code.co_name)
    model = Model(device="cuda", dtype=torch.float16)
    params = {}
    # more options for low GPU memory usage
    params.update({"chunk_size": 8})  # 8
    params.update({"merging_ratio": 0})  # 0
    # 320: bad quality
    params.update({"resolution": 320})  # 512

    prompt = 'oil painting of a deer, a high-quality, detailed, and professional photo'
    # video_path = '__assets__/depth_videos/deer.mp4'
    video_path = '__assets__/depth_videos/fox.mp4'
    out_path = f'./text2video_depth_control_{prompt}.mp4'
    model.process_controlnet_depth(
        video_path, prompt=prompt, save_path=out_path, **params
    )


# demo_t2v()
# demo_t2v_with_pose()
# demo_t2v_with_edge()
demo_t2v_with_edge_dreambooth()
# demo_t2v_with_depth()
