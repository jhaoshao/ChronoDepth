import argparse
import logging
import os
import random

from easydict import EasyDict
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import mediapy as media

from src.utils.video_utils import resize_max_res, colorize_video_depth
from chronodepth.unet_chronodepth import DiffusersUNetSpatioTemporalConditionModelChronodepth
from chronodepth.chronodepth_pipeline import ChronoDepthPipeline

def seed_all(seed: int = 0):
    """
    Set random seeds of all components.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def run_pipeline(pipe, cfg, video_rgb, generator, device):
    """
    Run the pipe on the input video.
    args:
        pipe: ChronoDepthPipeline object
        cfg: config object
        video_rgb: input video, torch.Tensor, shape [T, H, W, 3], range [0, 255]
        generator: torch.Generator
    returns:
        video_depth_pred: predicted depth, torch.Tensor, shape [T, H, W], range [0, 1]
    """
    if isinstance(video_rgb, torch.Tensor):
        video_rgb = video_rgb.cpu().numpy()

    original_height = video_rgb.shape[1]
    original_width = video_rgb.shape[2]

    # resize the video to the max resolution
    video_rgb = resize_max_res(video_rgb, cfg.max_res)

    video_rgb = video_rgb.astype(np.float32) / 255.0
    
    pipe_out = pipe(
        video_rgb,
        num_inference_steps=cfg.denoise_steps,
        decode_chunk_size=cfg.decode_chunk_size,
        motion_bucket_id=127,
        fps=7,
        noise_aug_strength=0.0,
        generator=generator,
        infer_mode=cfg.infer_mode,
        sigma_epsilon=cfg.sigma_epsilon,
    )

    depth_frames_pred = pipe_out.frames
    depth_frames_pred = torch.from_numpy(depth_frames_pred).to(device)
    depth_frames_pred = F.interpolate(depth_frames_pred, size=(original_height, original_width), mode="bilinear", align_corners=False)
    depth_frames_pred = depth_frames_pred.clamp(0, 1)
    depth_frames_pred = depth_frames_pred.squeeze(1)

    return depth_frames_pred


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run video depth estimation using ChronoDepth."
    )

    parser.add_argument(
        "--unet",
        type=str,
        default="jhshao/ChronoDepth-v1",
        help="Checkpoint path or hub name.",
    )
    parser.add_argument(
        "--model_base",
        type=str,
        default="stabilityai/stable-video-diffusion-img2vid-xt",
        help="Checkpoint path or hub name.",
    )

    # data setting
    parser.add_argument(
        "--data_dir", type=str, required=True, help="input data directory."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Whether save the output depth as grayscale.",
    )
    parser.add_argument(
        "--save_npy",
        action="store_true",
        help="Whether save the output depth as npy.",
    )

    # inference setting
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Max number of frames to process.",
    )
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=5,  # quantitative evaluation uses 5 steps
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed.",
    )
    parser.add_argument(
        "--infer_mode",
        type=str,
        default="ours",
        help="Inference mode, options: naive, replacement, ours",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=5, # quantitative evaluation uses 5
        help="Chunk size of sliding window for inference.",
    )
    parser.add_argument(
        "--n_tokens",
        type=int,
        default=10, # quantitative evaluation uses 10
        help="number of frames of each clip for sliding window inference.",
    )
    parser.add_argument(
        "--sigma_epsilon",
        type=float,
        default=-4.0, # quantitative evaluation uses -4.0
        help="hyperparameter for diffusion denoising.",
    )
    parser.add_argument(
        "--max_res",
        type=int,
        default=1024, # quantitative evaluation uses 1024
        help="Max resolution of the input video during inference.",
    )
    parser.add_argument(
        "--decode_chunk_size",
        type=int,
        default=8,
        help="Number of frames to decode per forward",
    )
    parser.add_argument(
        "--cpu_offload",
        type=str,
        default=None,
        help="Offload model to CPU to save memory, options: None, sequential, model",
    )
    
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    args = parser.parse_args()
    cfg = EasyDict(vars(args))

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
    unet = DiffusersUNetSpatioTemporalConditionModelChronodepth.from_pretrained(
            cfg.unet,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
    pipeline = ChronoDepthPipeline.from_pretrained(
                            cfg.model_base,
                            unet=unet,
                            torch_dtype=torch.float16,
                            variant="fp16",
                        )
    pipeline.n_tokens = cfg.n_tokens
    pipeline.chunk_size = cfg.chunk_size

    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except ImportError:
        logging.debug("run without xformers")

    # for saving memory, we can offload the model to CPU, or even run the model sequentially to save more memory
    if cfg.cpu_offload is not None:
        if cfg.cpu_offload == "sequential":
            # This will slow, but save more memory
            pipeline.enable_sequential_cpu_offload()
        elif cfg.cpu_offload == "model":
            pipeline.enable_model_cpu_offload()
        else:
            raise ValueError(f"Unknown cpu offload option: {cfg.cpu_offload}")
    else:
        pipeline.to(device)

    # -------------------- data --------------------
    video_name = cfg.data_dir.split('/')[-1].split('.')[0]
    video_data = media.read_video(cfg.data_dir)
    
    fps = video_data.metadata.fps
    video_rgb = np.array(video_data)
    if cfg.max_frames is not None:
        video_rgb = video_rgb[:cfg.max_frames]
    
    # -------------------- Inference and saving --------------------
    video_depth_pred = run_pipeline(pipeline, cfg, video_rgb, generator, device) # range [0, 1]

    if cfg.grayscale:
        colored_depth_video = video_depth_pred.cpu().numpy() * 255
        colored_depth_video = np.repeat(colored_depth_video[:, :, :, None], 3, axis=3)
        colored_depth_video = colored_depth_video.astype(np.uint8)
    else:
        colored_depth_video = colorize_video_depth(video_depth_pred)

    media.write_video(f"{cfg.output_dir}/{video_name}_depth.mp4", colored_depth_video, fps=fps)

    if cfg.save_npy:
        np.save(f"{cfg.output_dir}/{video_name}_depth.npy", video_depth_pred.cpu().numpy())