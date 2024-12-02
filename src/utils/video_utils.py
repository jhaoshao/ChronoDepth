import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch


def resize_max_res(video_rgb, max_res, interpolation=cv2.INTER_LINEAR):
    """
    Resize the video to the max resolution while keeping the aspect ratio.
    Args:
        video_rgb: (T, H, W, 3), RGB video, uint8
        max_res: int, max resolution
    Returns:
        video_rgb: (T, H_new, W_new, 3), resized RGB video, uint8
    """
    original_height = video_rgb.shape[1]
    original_width = video_rgb.shape[2]

    # round the height and width to the nearest multiple of 64
    height = round(original_height / 64) * 64
    width = round(original_width / 64) * 64

    # resize the video if the height or width is larger than max_res
    if max(height, width) > max_res:
        scale = max_res / max(original_height, original_width)
        height = round(original_height * scale / 64) * 64
        width = round(original_width * scale / 64) * 64

    frames = []
    for i in range(video_rgb.shape[0]):
        frames.append(cv2.resize(video_rgb[i], (width, height), interpolation=interpolation))

    frames = np.array(frames)
    return frames


def colorize_video_depth(depth_video, colormap="Spectral"):
    """
    Colorize the depth video using the specified colormap.
    depth_video: (T, H, W), depth video, [0, 1]
    return: 
    colored_depth_video: (T, H, W, 3), colored depth video, dtype=uint8
    """
    if isinstance(depth_video, torch.Tensor):
        depth_video = depth_video.cpu().numpy()
    T, H, W = depth_video.shape
    colored_depth_video = []
    for i in range(T):
        colored_depth = plt.get_cmap(colormap)(depth_video[i], bytes=True)[...,:3]
        colored_depth_video.append(colored_depth)
    colored_depth_video = np.stack(colored_depth_video, axis=0)
    
    return colored_depth_video