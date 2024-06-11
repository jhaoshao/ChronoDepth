import argparse
import logging
import os
import random

from easydict import EasyDict
import numpy as np
import torch
from PIL import Image
import mediapy as media
from tqdm.auto import tqdm

from chronodepth.chronodepth_pipeline import ChronoDepthPipeline

def seed_all(seed: int = 0):
    """
    Set random seeds of all components.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_video(video_path):
    return media.read_video(video_path)

def export_to_video(video_frames, output_video_path, fps):
    media.write_video(output_video_path, video_frames, fps=fps)

if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using Marigold."
    )

    parser.add_argument(
        "--model_base",
        type=str,
        default="jhshao/ChronoDepth",
        help="Checkpoint path or hub name.",
    )

    # data setting
    parser.add_argument(
        "--data_dir", type=str, required=True, help="input data directory."
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )

    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=10,  # quantitative evaluation uses 50 steps
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed.",
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=1,
        help="Number of frames to infer per forward",
    )
    parser.add_argument(
        "--decode_chunk_size",
        type=int,
        default=1,
        help="Number of frames to decode per forward",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
        help="Window size for inpaint inference",
    )
    
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    args = parser.parse_args()
    cfg = EasyDict(vars(args))

    if cfg.window_size is None or cfg.window_size == cfg.num_frames:
        cfg.inpaint_inference = False
    else:
        cfg.inpaint_inference = True

    print(cfg)

    # -------------------- Preparation --------------------
    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Random Seed --------------------
    if cfg.seed is None:
        import time
        cfg.seed = int(time.time())
    seed_all(cfg.seed)

    generator = torch.Generator(
        device=device).manual_seed(cfg.seed)
    
    assert cfg.data_dir.endswith(".mp4"), "data_dir should be mp4 file."
    os.makedirs(cfg.output_dir, exist_ok=True)
    logging.info(f"output dir = {cfg.output_dir}")

    # -------------------- Model --------------------
    if cfg.half_precision:
        weight_dtype = torch.float16
        logging.info(f"Running with half precision ({weight_dtype}).")
    else:
        weight_dtype = torch.float32

    pipeline = ChronoDepthPipeline.from_pretrained(
                            cfg.model_base,
                            torch_dtype=weight_dtype,
                        )

    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except ImportError:
        logging.debug("run without xformers")

    pipe = pipeline.to(device)

    # -------------------- data --------------------
    data_ls = []
    video_data = read_video(cfg.data_dir)
    fps = video_data.metadata.fps
    for i in tqdm(range(len(video_data)-cfg.num_frames+1)):
        is_first_clip = i == 0
        is_last_clip = i == len(video_data) - cfg.num_frames
        is_new_clip = (
            (cfg.inpaint_inference and i % cfg.window_size == 0)
            or (cfg.inpaint_inference == False and i % cfg.num_frames == 0)
        )
        if is_first_clip or is_last_clip or is_new_clip:
            data_ls.append(np.array(video_data[i: i+cfg.num_frames])) # [t, H, W, 3]

    video_name = cfg.data_dir.split('/')[-1].split('.')[0]
    video_length = len(video_data)
    
    depth_colored_pred = []
    depth_pred = []
    rgb_int_ls = []
    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        for iter, batch in enumerate(tqdm(
            data_ls, desc=f"Inferencing on {cfg.data_dir}", leave=True
        )):
            rgb_int = batch
            input_images = [Image.fromarray(rgb_int[i]) for i in range(cfg.num_frames)]

            # Predict depth
            if iter == 0: # First clip:
                pipe_out = pipeline(
                    input_images,
                    num_frames=len(input_images),
                    num_inference_steps=cfg.denoise_steps,
                    decode_chunk_size=cfg.decode_chunk_size,
                    motion_bucket_id=127,
                    fps=7,
                    noise_aug_strength=0.0,
                    generator=generator,
                )
            elif cfg.inpaint_inference and (iter == len(data_ls) - 1): # temporal inpaint inference for last clip
                last_window_size = cfg.window_size if video_length%cfg.window_size == 0 else video_length%cfg.window_size
                pipe_out = pipeline(
                    input_images,
                    num_frames=cfg.num_frames,
                    num_inference_steps=cfg.denoise_steps,
                    decode_chunk_size=cfg.decode_chunk_size,
                    motion_bucket_id=127,
                    fps=7,
                    noise_aug_strength=0.0,
                    generator=generator,
                    depth_pred_last=depth_frames_pred_ts[last_window_size:],
                )
            elif cfg.inpaint_inference and iter > 0: # temporal inpaint inference
                pipe_out = pipeline(
                    input_images,
                    num_frames=cfg.num_frames,
                    num_inference_steps=cfg.denoise_steps,
                    decode_chunk_size=cfg.decode_chunk_size,
                    motion_bucket_id=127,
                    fps=7,
                    noise_aug_strength=0.0,
                    generator=generator,
                    depth_pred_last=depth_frames_pred_ts[cfg.window_size:],
                )
            else: # Separate inference
                pipe_out = pipeline(
                    input_images,
                    num_frames=cfg.num_frames,
                    num_inference_steps=cfg.denoise_steps,
                    decode_chunk_size=cfg.decode_chunk_size,
                    motion_bucket_id=127,
                    fps=7,
                    noise_aug_strength=0.0,
                    generator=generator,
                )

            depth_frames_pred = [pipe_out.depth_np[i] for i in range(cfg.num_frames)]

            depth_frames_colored_pred = []
            for i in range(cfg.num_frames):
                depth_frame_colored_pred = np.array(pipe_out.depth_colored[i])
                depth_frames_colored_pred.append(depth_frame_colored_pred)
            depth_frames_colored_pred = np.stack(depth_frames_colored_pred, axis=0)

            depth_frames_pred = np.stack(depth_frames_pred, axis=0)
            depth_frames_pred_ts = torch.from_numpy(depth_frames_pred).to(device)
            depth_frames_pred_ts = depth_frames_pred_ts * 2 - 1

            if cfg.inpaint_inference == False:
                if iter == len(data_ls) - 1:
                    last_window_size = cfg.num_frames if video_length%cfg.num_frames == 0 else video_length%cfg.num_frames
                    rgb_int_ls.append(rgb_int[-last_window_size:])
                    depth_colored_pred.append(depth_frames_colored_pred[-last_window_size:])
                    depth_pred.append(depth_frames_pred[-last_window_size:])
                else:
                    rgb_int_ls.append(rgb_int)
                    depth_colored_pred.append(depth_frames_colored_pred)
                    depth_pred.append(depth_frames_pred)
            else:
                if iter == 0:
                    rgb_int_ls.append(rgb_int)
                    depth_colored_pred.append(depth_frames_colored_pred)
                    depth_pred.append(depth_frames_pred)
                elif iter == len(data_ls) - 1:
                    rgb_int_ls.append(rgb_int[-last_window_size:])
                    depth_colored_pred.append(depth_frames_colored_pred[-last_window_size:])
                    depth_pred.append(depth_frames_pred[-last_window_size:])
                else:
                    rgb_int_ls.append(rgb_int[-cfg.window_size:])
                    depth_colored_pred.append(depth_frames_colored_pred[-cfg.window_size:])
                    depth_pred.append(depth_frames_pred[-cfg.window_size:])

    rgb_int_ls = np.concatenate(rgb_int_ls, axis=0)
    depth_colored_pred = np.concatenate(depth_colored_pred, axis=0)
    depth_pred = np.concatenate(depth_pred, axis=0)

    # -------------------- Save results --------------------
    output_dir_video = os.path.join(cfg.output_dir, video_name)
    os.makedirs(output_dir_video, exist_ok=True)

    # # Save images
    # rgb_dir = os.path.join(output_dir_video, "rgb")
    # depth_colored_dir = os.path.join(output_dir_video, "depth_colored")
    # depth_pred_dir = os.path.join(output_dir_video, "depth_pred")
    # os.makedirs(rgb_dir, exist_ok=True)
    # os.makedirs(depth_colored_dir, exist_ok=True)
    # os.makedirs(depth_pred_dir, exist_ok=True)
    # for i in tqdm(range(len(rgb_int_ls))):
    #     Image.fromarray(rgb_int_ls[i]).save(os.path.join(rgb_dir, f"frame_{i:06d}.png"))     
    #     Image.fromarray(depth_colored_pred[i]).save(os.path.join(depth_colored_dir, f"frame_{i:06d}.png"))
    #     np.save(os.path.join(depth_pred_dir, f"frame_{i:06d}.npy"), depth_pred[i])

    # Export to video
    export_to_video(rgb_int_ls, os.path.join(output_dir_video , f"{video_name}_rgb_clipFrames{cfg.num_frames}_ws{cfg.window_size}.mp4"), fps)
    export_to_video(depth_colored_pred, os.path.join(output_dir_video , f"{video_name}_depth_clipFrame{cfg.num_frames}_ws{cfg.window_size}.mp4"), fps)