# Adapted from Marigold: https://github.com/prs-eth/Marigold and diffusers

import inspect
from typing import Union, Optional, List

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import PIL
from PIL import Image
from diffusers import (
    DiffusionPipeline,
    EulerDiscreteScheduler,
    UNetSpatioTemporalConditionModel,
    AutoencoderKLTemporalDecoder,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from transformers import (
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)
from einops import rearrange, repeat


class ChronoDepthOutput(BaseOutput):
    r"""
    Output class for zero-shot text-to-video pipeline.

    Args:
        frames (`[List[PIL.Image.Image]`, `np.ndarray`]):
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
    """
    depth_np: np.ndarray
    depth_colored: Union[List[PIL.Image.Image], np.ndarray]


class ChronoDepthPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]
    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215

    def __init__(
        self,
        vae: AutoencoderKLTemporalDecoder,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNetSpatioTemporalConditionModel,
        scheduler: EulerDiscreteScheduler,
        feature_extractor: CLIPImageProcessor,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        if not hasattr(self, "dtype"):
            self.dtype = self.unet.dtype

    def encode_RGB(self,
                   image: torch.Tensor,
                   ):
        video_length = image.shape[1]
        image = rearrange(image, "b f c h w -> (b f) c h w")
        latents = self.vae.encode(image).latent_dist.sample()
        latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
        latents = latents * self.vae.config.scaling_factor
        
        return latents
    
    def _encode_image(self, image, device, discard=True):
        '''
        set image to zero tensor discards the image embeddings if discard is True
        '''
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.image_processor.pil_to_numpy(image)
            if discard:
                image = np.zeros_like(image)
            image = self.image_processor.numpy_to_pt(image)

            # We normalize the image before resizing to match with the original implementation.
            # Then we unnormalize it after resizing.
            image = image * 2.0 - 1.0
            image = _resize_with_antialiasing(image, (224, 224))
            image = (image + 1.0) / 2.0

            # Normalize the image with for CLIP input
            image = self.feature_extractor(
                images=image,
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

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

    def _get_add_time_ids(self,
                          fps,
                          motion_bucket_id,
                          noise_aug_strength,
                          dtype,
                          batch_size,
                          ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * \
            len(add_time_ids)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size, 1)
        return add_time_ids

    def decode_latents(self, latents, num_frames, decode_chunk_size=14):
        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
        latents = latents.flatten(0, 1)

        latents = 1 / self.vae.config.scaling_factor * latents

        forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
        accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())

        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                # we only pass num_frames_in if it's expected
                decode_kwargs["num_frames"] = num_frames_in

            frame = self.vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)

        # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.float()
        return frames

    def check_inputs(self, image, height, width):
        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

        if height % 64 != 0 or width % 64 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    def prepare_latents(
        self,
        shape,
        dtype,
        device,
        generator,
        latent=None,
    ):
        if isinstance(generator, list) and len(generator) != shape[0]:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {shape[0]}. Make sure the batch size matches the length of the generators."
            )

        if latent is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    def __call__(
        self,
        input_image: Union[List[PIL.Image.Image], torch.FloatTensor],
        height: int = 576,
        width: int = 768,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 10,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: Optional[int] = None,
        color_map: str="Spectral",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        show_progress_bar: bool = True,
        match_input_res: bool = True,
        depth_pred_last: Optional[torch.FloatTensor] = None,
    ):
        assert height >= 0 and width >=0
        assert num_inference_steps >=1

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(input_image, height, width)

        # 2. Define call parameters
        if isinstance(input_image, list):
            batch_size = 1
            input_size = input_image[0].size
        elif isinstance(input_image, torch.Tensor):
            batch_size = input_image.shape[0]
            input_size = input_image.shape[:-3:-1]
        assert batch_size == 1, "Batch size must be 1 for now"
        device = self._execution_device

        # 3. Encode input image
        image_embeddings = self._encode_image(input_image[0], device)
        image_embeddings = image_embeddings.repeat((batch_size, 1, 1))

        # NOTE: Stable Diffusion Video was conditioned on fps - 1, which
        # is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        fps = fps - 1

        # 4. Encode input image using VAE
        input_image = self.image_processor.preprocess(input_image, height=height, width=width).to(device)
        assert input_image.min() >= -1.0 and input_image.max() <= 1.0
        noise = randn_tensor(input_image.shape, generator=generator, device=device, dtype=input_image.dtype)
        input_image = input_image + noise_aug_strength * noise
        if depth_pred_last is not None:
            depth_pred_last = depth_pred_last.to(device)
            # resize depth
            from torchvision.transforms import InterpolationMode
            from torchvision.transforms.functional import resize
            depth_pred_last = resize(depth_pred_last.unsqueeze(1), (height, width), InterpolationMode.NEAREST_EXACT, antialias=True)
            depth_pred_last = repeat(depth_pred_last, 'f c h w ->b f c h w', b=batch_size)

        rgb_batch = repeat(input_image, 'f c h w ->b f c h w', b=batch_size)

        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
        )
        added_time_ids = added_time_ids.to(device)

        depth_pred_raw = self.single_infer(rgb_batch, 
                                           image_embeddings,
                                           added_time_ids,
                                           num_inference_steps,
                                           show_progress_bar,
                                           generator,
                                           depth_pred_last=depth_pred_last,
                                           decode_chunk_size=decode_chunk_size)
        
        depth_colored_img_list = []
        depth_frames = []
        for i in range(num_frames):
            depth_frame = depth_pred_raw[:, i].squeeze()
        
            # Convert to numpy
            depth_frame = depth_frame.cpu().numpy().astype(np.float32)

            if match_input_res:
                pred_img = Image.fromarray(depth_frame)
                pred_img = pred_img.resize(input_size, resample=Image.NEAREST)
                depth_frame = np.asarray(pred_img)

            # Clip output range: current size is the original size
            depth_frame = depth_frame.clip(0, 1)
        
            # Colorize
            depth_colored = plt.get_cmap(color_map)(depth_frame, bytes=True)[..., :3]
            depth_colored_img = Image.fromarray(depth_colored)
            
            depth_colored_img_list.append(depth_colored_img)
            depth_frames.append(depth_frame)
        
        depth_frame = np.stack(depth_frames)

        self.maybe_free_model_hooks()

        return ChronoDepthOutput(
            depth_np = depth_frames,
            depth_colored = depth_colored_img_list,
        )

    @torch.no_grad()
    def single_infer(self,
                     input_rgb: torch.Tensor,
                     image_embeddings: torch.Tensor,
                     added_time_ids: torch.Tensor,
                     num_inference_steps: int,
                     show_pbar: bool,
                     generator: Optional[Union[torch.Generator, List[torch.Generator]]],
                     depth_pred_last: Optional[torch.Tensor] = None,
                     decode_chunk_size=1,
                     ):
        device = input_rgb.device

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        rgb_latent = self.encode_RGB(input_rgb)
        rgb_latent = rgb_latent.to(image_embeddings.dtype)
        if depth_pred_last is not None:
            depth_pred_last = depth_pred_last.repeat(1, 1, 3, 1, 1)
            depth_pred_last_latent = self.encode_RGB(depth_pred_last)
            depth_pred_last_latent = depth_pred_last_latent.to(image_embeddings.dtype)
        else:
            depth_pred_last_latent = None
        
        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        depth_latent = self.prepare_latents(
            rgb_latent.shape,
            image_embeddings.dtype,
            device,
            generator
        )

        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)
        
        for i, t in iterable:
            if depth_pred_last_latent is not None:
                known_frames_num = depth_pred_last_latent.shape[1]
                epsilon = randn_tensor(
                    depth_pred_last_latent.shape, 
                    generator=generator, 
                    device=device, 
                    dtype=image_embeddings.dtype
                    )
                depth_latent[:, :known_frames_num] = depth_pred_last_latent + epsilon * self.scheduler.sigmas[i]
            depth_latent = self.scheduler.scale_model_input(depth_latent, t)
            unet_input = torch.cat([rgb_latent, depth_latent], dim=2)
            
            noise_pred = self.unet(
                unet_input, t, image_embeddings, added_time_ids=added_time_ids
            )[0]

            # compute the previous noisy sample x_t -> x_t-1
            depth_latent = self.scheduler.step(noise_pred, t, depth_latent).prev_sample
        
        torch.cuda.empty_cache()
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
        depth = self.decode_depth(depth_latent, decode_chunk_size=decode_chunk_size)
        # clip prediction
        depth = torch.clip(depth, -1.0, 1.0)
        # shift to [0, 1]
        depth = (depth + 1.0) / 2.0

        return depth
    
# resizing utils
def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out