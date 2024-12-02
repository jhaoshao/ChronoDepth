import inspect
from typing import Union, Optional, List

import torch
import numpy as np
from tqdm.auto import tqdm
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    _resize_with_antialiasing,
    StableVideoDiffusionPipelineOutput,
    StableVideoDiffusionPipeline,
)
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from einops import rearrange


class ChronoDepthPipeline(StableVideoDiffusionPipeline):

    @torch.inference_mode()
    def encode_images(self,
                   images: torch.Tensor,
                   decode_chunk_size=5,
                   ):
        video_length = images.shape[1]
        images = rearrange(images, "b f c h w -> (b f) c h w")
        latents = []
        for i in range(0, images.shape[0], decode_chunk_size):
            latents_chunk = self.vae.encode(images[i : i + decode_chunk_size]).latent_dist.sample()
            latents.append(latents_chunk)
        latents = torch.cat(latents, dim=0)
        latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
        latents = latents * self.vae.config.scaling_factor
        return latents
    
    @torch.inference_mode()
    def _encode_image(self, images, device, discard=True, chunk_size=14):
        '''
        set image to zero tensor discards the image embeddings if discard is True
        '''
        dtype = next(self.image_encoder.parameters()).dtype

        images = _resize_with_antialiasing(images, (224, 224))
        images = (images + 1.0) / 2.0
        
        if discard:
            images = torch.zeros_like(images)

        image_embeddings = []
        for i in range(0, images.shape[0], chunk_size):
            tmp = self.feature_extractor(
                images=images[i : i + chunk_size],
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values

            tmp = tmp.to(device=device, dtype=dtype)
            image_embeddings.append(self.image_encoder(tmp).image_embeds)
        image_embeddings = torch.cat(image_embeddings, dim=0)
        image_embeddings = image_embeddings.unsqueeze(1) # [t, 1, 1024]

        return image_embeddings
    
    def decode_depth(self, depth_latent: torch.Tensor, decode_chunk_size=5) -> torch.Tensor:
        num_frames = depth_latent.shape[1]
        depth_latent = rearrange(depth_latent, "b f c h w -> (b f) c h w")

        depth_latent = depth_latent / self.vae.config.scaling_factor

        forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
        accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())
        
        depth_frames = []
        for i in range(0, depth_latent.shape[0], decode_chunk_size):
            num_frames_in = depth_latent[i : i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                # we only pass num_frames_in if it's expected
                decode_kwargs["num_frames"] = num_frames_in

            depth_frame = self.vae.decode(depth_latent[i : i + decode_chunk_size], **decode_kwargs).sample
            depth_frames.append(depth_frame)

        depth_frames = torch.cat(depth_frames, dim=0)
        depth_frames = depth_frames.reshape(-1, num_frames, *depth_frames.shape[1:])
        depth_mean = depth_frames.mean(dim=2, keepdim=True)
        
        return depth_mean

    @staticmethod
    def check_inputs(images, height, width):
        if (
            not isinstance(images, torch.Tensor)
            and not isinstance(images, np.ndarray)
        ):
            raise ValueError(
                "`images` has to be of type `torch.Tensor` or `numpy.ndarray` but is"
                f" {type(images)}"
            )

        if height % 64 != 0 or width % 64 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    @torch.no_grad()
    def __call__(
        self,
        input_images: Union[np.ndarray, torch.FloatTensor],
        height: int = 576,
        width: int = 768,
        num_inference_steps: int = 10,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: Optional[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        show_progress_bar: bool = True,
        latents: Optional[torch.Tensor] = None,
        infer_mode: str = 'ours',
        sigma_epsilon: float = -4,
    ):
        """
        Args:
            input_images: shape [T, H, W, 3] if np.ndarray or [T, 3, H, W] if torch.FloatTensor, range [0, 1]
            height: int, height of the input image
            width: int, width of the input image
            num_inference_steps: int, number of inference steps
            fps: int, frames per second
            motion_bucket_id: int, motion bucket id
            noise_aug_strength: float, noise augmentation strength
            decode_chunk_size: int, decode chunk size
            generator: torch.Generator or List[torch.Generator], random number generator
            show_progress_bar: bool, show progress bar
        """
        assert height >= 0 and width >=0
        assert num_inference_steps >=1

        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else 8

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(input_images, height, width)

        # 2. Define call parameters
        batch_size = 1 # only support batch size 1 for now
        device = self._execution_device

        # 3. Encode input image
        if isinstance(input_images, np.ndarray):
            input_images = torch.from_numpy(input_images.transpose(0, 3, 1, 2))
        else:
            assert isinstance(input_images, torch.Tensor)
        input_images = input_images.to(device=device)
        input_images = input_images * 2.0 - 1.0  # [0,1] -> [-1,1], in [t, c, h, w]
        
        discard_clip_features = True
        image_embeddings = self._encode_image(input_images, device, 
                                              discard=discard_clip_features,
                                              chunk_size=decode_chunk_size
                                              )

        # NOTE: Stable Diffusion Video was conditioned on fps - 1, which
        # is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        fps = fps - 1

        # 4. Encode input image using VAE
        noise = randn_tensor(input_images.shape, generator=generator, device=device, dtype=input_images.dtype)
        input_images = input_images + noise_aug_strength * noise
        
        rgb_batch = input_images.unsqueeze(0)

        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            1, # do not modify this! 
            False, # do not modify this! 
        )
        added_time_ids = added_time_ids.to(device)

        if infer_mode == 'ours':
            depth_pred_raw = self.single_infer_ours(
                rgb_batch, 
                image_embeddings,
                added_time_ids,
                num_inference_steps,
                show_progress_bar,
                generator,
                decode_chunk_size=decode_chunk_size,
                latents=latents,
                sigma_epsilon=sigma_epsilon,
            )
        elif infer_mode == 'replacement':
            depth_pred_raw = self.single_infer_replacement(
                rgb_batch, 
                image_embeddings,
                added_time_ids,
                num_inference_steps,
                show_progress_bar,
                generator,
                decode_chunk_size=decode_chunk_size,
                latents=latents,
            )
        elif infer_mode == 'naive':
            depth_pred_raw = self.single_infer_naive_sliding_window(
                rgb_batch, 
                image_embeddings,
                added_time_ids,
                num_inference_steps,
                show_progress_bar,
                generator,
                decode_chunk_size=decode_chunk_size,
                latents=latents,
            )

        
        depth_frames = depth_pred_raw.cpu().numpy().astype(np.float32)

        self.maybe_free_model_hooks()

        return StableVideoDiffusionPipelineOutput(
            frames = depth_frames,
        )

    @torch.no_grad()
    def single_infer_ours(self,
                     input_rgb: torch.Tensor,
                     image_embeddings: torch.Tensor,
                     added_time_ids: torch.Tensor,
                     num_inference_steps: int,
                     show_pbar: bool,
                     generator: Optional[Union[torch.Generator, List[torch.Generator]]],
                     decode_chunk_size=1,
                     latents: Optional[torch.Tensor] = None,
                     sigma_epsilon: float = -4,
                     ):
        device = input_rgb.device
        H, W = input_rgb.shape[-2:]

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        rgb_latent = self.encode_images(input_rgb)
        rgb_latent = rgb_latent.to(image_embeddings.dtype)
        
        torch.cuda.empty_cache()
        
        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        batch_size, n_frames, _, _, _ = rgb_latent.shape
        num_channels_latents = self.unet.config.in_channels

        curr_frame = 0
        depth_latent = torch.tensor([], dtype=image_embeddings.dtype, device=device)
        pbar = tqdm(total=n_frames, initial=curr_frame, desc="Sampling")

        # first chunk
        horizon = min(n_frames-curr_frame, self.n_tokens)
        start_frame = 0
        chunk = self.prepare_latents(
                batch_size,
                horizon,
                num_channels_latents,
                H,
                W,
                image_embeddings.dtype,
                device,
                generator,
                latents,
            )
        depth_latent = torch.cat([depth_latent, chunk], 1)
        if show_pbar:
                iterable = tqdm(
                    enumerate(timesteps),
                    total=len(timesteps),
                    leave=False,
                    desc=" " * 4 + "Diffusion denoising first chunk",
                )   
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            curr_timesteps = torch.tensor([t]*horizon).to(device)
            depth_latent = self.scheduler.scale_model_input(depth_latent, t)
            noise_pred = self.unet(
                torch.cat([rgb_latent[:, start_frame:curr_frame+horizon], depth_latent[:, start_frame:]], dim=2), 
                curr_timesteps[start_frame:],
                image_embeddings[start_frame:curr_frame+horizon],
                added_time_ids=added_time_ids
            )[0]
            depth_latent[:, curr_frame:] = self.scheduler.step(noise_pred[:,-horizon:], t, depth_latent[:, curr_frame:]).prev_sample
        
        self.scheduler._step_index = None
        curr_frame += horizon
        pbar.update(horizon)
        
        while curr_frame < n_frames:
            if self.chunk_size > 0:
                horizon = min(n_frames - curr_frame, self.chunk_size)
            else:
                horizon = min(n_frames - curr_frame, self.n_tokens)
            assert horizon <= self.n_tokens, "horizon exceeds the number of tokens."
            chunk = self.prepare_latents(
                batch_size, 
                horizon, 
                num_channels_latents,
                H,
                W,
                image_embeddings.dtype,
                device,
                generator,
                latents,
            )
            depth_latent = torch.cat([depth_latent, chunk], 1)
            start_frame = max(0, curr_frame + horizon - self.n_tokens)

            pbar.set_postfix(
                {
                    "start": start_frame,
                    "end": curr_frame + horizon,
                }
            )

            if show_pbar:
                iterable = tqdm(
                    enumerate(timesteps),
                    total=len(timesteps),
                    leave=False,
                    desc=" " * 4 + "Diffusion denoising ",
                )   
            else:
                iterable = enumerate(timesteps)

            for i, t in iterable:
                t_horizon = torch.tensor([t]*horizon).to(device)
                # t_context = timesteps[-1] * torch.ones((curr_frame,), dtype=t.dtype).to(device)
                t_context = sigma_epsilon * torch.ones((curr_frame,), dtype=t.dtype).to(device)
                curr_timesteps = torch.concatenate((t_context, t_horizon), 0)
                depth_latent[:, curr_frame:] = self.scheduler.scale_model_input(depth_latent[:, curr_frame:], t)
                noise_pred = self.unet(
                    torch.cat([rgb_latent[:, start_frame:curr_frame+horizon], depth_latent[:, start_frame:]], dim=2), 
                    curr_timesteps[start_frame:],
                    image_embeddings[start_frame:curr_frame+horizon],
                    added_time_ids=added_time_ids
                )[0]
                depth_latent[:, curr_frame:] = self.scheduler.step(noise_pred[:,-horizon:], t, depth_latent[:, curr_frame:]).prev_sample
            
            self.scheduler._step_index = None
            curr_frame += horizon
            pbar.update(horizon)

        torch.cuda.empty_cache()
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
        depth = self.decode_depth(depth_latent, decode_chunk_size=decode_chunk_size)
        # clip prediction
        depth = torch.clip(depth, -1.0, 1.0)
        # shift to [0, 1]
        depth = (depth + 1.0) / 2.0

        return depth.squeeze(0)
    
    @torch.no_grad()
    def single_infer_replacement(self,
                     input_rgb: torch.Tensor,
                     image_embeddings: torch.Tensor,
                     added_time_ids: torch.Tensor,
                     num_inference_steps: int,
                     show_pbar: bool,
                     generator: Optional[Union[torch.Generator, List[torch.Generator]]],
                     decode_chunk_size=1,
                     latents: Optional[torch.Tensor] = None,
                     ):
        device = input_rgb.device
        H, W = input_rgb.shape[-2:]

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        rgb_latent = self.encode_images(input_rgb)
        rgb_latent = rgb_latent.to(image_embeddings.dtype)
        
        torch.cuda.empty_cache()
        
        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        batch_size, n_frames, _, _, _ = rgb_latent.shape
        num_channels_latents = self.unet.config.in_channels

        curr_frame = 0
        depth_latent = torch.tensor([], dtype=image_embeddings.dtype, device=device)
        pbar = tqdm(total=n_frames, initial=curr_frame, desc="Sampling")

        # first chunk
        horizon = min(n_frames-curr_frame, self.n_tokens)
        start_frame = 0
        chunk = self.prepare_latents(
                batch_size,
                horizon,
                num_channels_latents,
                H,
                W,
                image_embeddings.dtype,
                device,
                generator,
                latents,
            )
        depth_latent = torch.cat([depth_latent, chunk], 1)
        if show_pbar:
                iterable = tqdm(
                    enumerate(timesteps),
                    total=len(timesteps),
                    leave=False,
                    desc=" " * 4 + "Diffusion denoising first chunk",
                )   
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            curr_timesteps = torch.tensor([t]*horizon).to(device)
            depth_latent = self.scheduler.scale_model_input(depth_latent, t)
            noise_pred = self.unet(
                torch.cat([rgb_latent[:, start_frame:curr_frame+horizon], depth_latent[:, start_frame:]], dim=2), 
                curr_timesteps[start_frame:],
                image_embeddings[start_frame:curr_frame+horizon],
                added_time_ids=added_time_ids
            )[0]
            depth_latent[:, curr_frame:] = self.scheduler.step(noise_pred[:,-horizon:], t, depth_latent[:, curr_frame:]).prev_sample
        
        self.scheduler._step_index = None
        curr_frame += horizon
        pbar.update(horizon)
        
        while curr_frame < n_frames:
            if self.chunk_size > 0:
                horizon = min(n_frames - curr_frame, self.chunk_size)
            else:
                horizon = min(n_frames - curr_frame, self.n_tokens)
            assert horizon <= self.n_tokens, "horizon exceeds the number of tokens."
            chunk = self.prepare_latents(
                batch_size, 
                horizon, 
                num_channels_latents,
                H,
                W,
                image_embeddings.dtype,
                device,
                generator,
                latents,
            )
            depth_latent = torch.cat([depth_latent, chunk], 1)
            start_frame = max(0, curr_frame + horizon - self.n_tokens)
            depth_pred_last_latent = depth_latent[:, start_frame:curr_frame].clone()

            pbar.set_postfix(
                {
                    "start": start_frame,
                    "end": curr_frame + horizon,
                }
            )

            if show_pbar:
                iterable = tqdm(
                    enumerate(timesteps),
                    total=len(timesteps),
                    leave=False,
                    desc=" " * 4 + "Diffusion denoising ",
                )   
            else:
                iterable = enumerate(timesteps)

            for i, t in iterable:
                curr_timesteps = torch.tensor([t]*(curr_frame+horizon-start_frame)).to(device)
                epsilon = randn_tensor(
                    depth_pred_last_latent.shape, 
                    generator=generator, 
                    device=device, 
                    dtype=image_embeddings.dtype
                )
                depth_latent[:, start_frame:curr_frame] = depth_pred_last_latent + epsilon * self.scheduler.sigmas[i]
                depth_latent[:, start_frame:] = self.scheduler.scale_model_input(depth_latent[:, start_frame:], t)
                noise_pred = self.unet(
                    torch.cat([rgb_latent[:, start_frame:curr_frame+horizon], depth_latent[:, start_frame:]], dim=2), 
                    curr_timesteps,
                    image_embeddings[start_frame:curr_frame+horizon],
                    added_time_ids=added_time_ids
                )[0]
                depth_latent[:, start_frame:] = self.scheduler.step(noise_pred, t, depth_latent[:, start_frame:]).prev_sample
            
            depth_latent[:, start_frame:curr_frame] = depth_pred_last_latent
            self.scheduler._step_index = None
            curr_frame += horizon
            pbar.update(horizon)

        torch.cuda.empty_cache()
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
        depth = self.decode_depth(depth_latent, decode_chunk_size=decode_chunk_size)
        # clip prediction
        depth = torch.clip(depth, -1.0, 1.0)
        # shift to [0, 1]
        depth = (depth + 1.0) / 2.0

        return depth.squeeze(0)
    
    @torch.no_grad()
    def single_infer_naive_sliding_window(self,
                     input_rgb: torch.Tensor,
                     image_embeddings: torch.Tensor,
                     added_time_ids: torch.Tensor,
                     num_inference_steps: int,
                     show_pbar: bool,
                     generator: Optional[Union[torch.Generator, List[torch.Generator]]],
                     decode_chunk_size=1,
                     latents: Optional[torch.Tensor] = None,
                     ):
        device = input_rgb.device
        H, W = input_rgb.shape[-2:]

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        rgb_latent = self.encode_images(input_rgb)
        rgb_latent = rgb_latent.to(image_embeddings.dtype)
        
        torch.cuda.empty_cache()
        
        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        batch_size, n_frames, _, _, _ = rgb_latent.shape
        num_channels_latents = self.unet.config.in_channels

        curr_frame = 0
        depth_latent = torch.tensor([], dtype=image_embeddings.dtype, device=device)
        pbar = tqdm(total=n_frames, initial=curr_frame, desc="Sampling")

        # first chunk
        horizon = min(n_frames-curr_frame, self.n_tokens)
        start_frame = 0
        chunk = self.prepare_latents(
                batch_size,
                horizon,
                num_channels_latents,
                H,
                W,
                image_embeddings.dtype,
                device,
                generator,
                latents,
            )
        depth_latent = torch.cat([depth_latent, chunk], 1)
        if show_pbar:
                iterable = tqdm(
                    enumerate(timesteps),
                    total=len(timesteps),
                    leave=False,
                    desc=" " * 4 + "Diffusion denoising first chunk",
                )   
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            curr_timesteps = torch.tensor([t]*horizon).to(device)
            depth_latent = self.scheduler.scale_model_input(depth_latent, t)
            noise_pred = self.unet(
                torch.cat([rgb_latent[:, start_frame:curr_frame+horizon], depth_latent[:, start_frame:]], dim=2), 
                curr_timesteps[start_frame:],
                image_embeddings[start_frame:curr_frame+horizon],
                added_time_ids=added_time_ids
            )[0]
            depth_latent[:, curr_frame:] = self.scheduler.step(noise_pred[:,-horizon:], t, depth_latent[:, curr_frame:]).prev_sample
        
        self.scheduler._step_index = None
        curr_frame += horizon
        pbar.update(horizon)
        
        while curr_frame < n_frames:
            if self.chunk_size > 0:
                horizon = min(n_frames - curr_frame, self.chunk_size)
            else:
                horizon = min(n_frames - curr_frame, self.n_tokens)
            assert horizon <= self.n_tokens, "horizon exceeds the number of tokens."
            start_frame = max(0, curr_frame + horizon - self.n_tokens)

            chunk = self.prepare_latents(
                batch_size, 
                curr_frame+horizon-start_frame, 
                num_channels_latents,
                H,
                W,
                image_embeddings.dtype,
                device,
                generator,
                latents,
            )

            pbar.set_postfix(
                {
                    "start": start_frame,
                    "end": curr_frame + horizon,
                }
            )

            if show_pbar:
                iterable = tqdm(
                    enumerate(timesteps),
                    total=len(timesteps),
                    leave=False,
                    desc=" " * 4 + "Diffusion denoising ",
                )   
            else:
                iterable = enumerate(timesteps)

            for i, t in iterable:
                curr_timesteps = torch.tensor([t]*(curr_frame+horizon-start_frame)).to(device)
                chunk = self.scheduler.scale_model_input(chunk, t)
                noise_pred = self.unet(
                    torch.cat([rgb_latent[:, start_frame:curr_frame+horizon], chunk], dim=2), 
                    curr_timesteps,
                    image_embeddings[start_frame:curr_frame+horizon],
                    added_time_ids=added_time_ids
                )[0]
                chunk = self.scheduler.step(noise_pred, t, chunk).prev_sample
            
            depth_latent = torch.cat([depth_latent, chunk[:, -horizon:]], 1)
            self.scheduler._step_index = None
            curr_frame += horizon
            pbar.update(horizon)

        torch.cuda.empty_cache()
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
        depth = self.decode_depth(depth_latent, decode_chunk_size=decode_chunk_size)
        # clip prediction
        depth = torch.clip(depth, -1.0, 1.0)
        # shift to [0, 1]
        depth = (depth + 1.0) / 2.0

        return depth.squeeze(0)