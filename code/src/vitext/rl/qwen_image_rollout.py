"""Sinh ảnh rollout cho RL bằng chính transformer đang train (adapter "old"), không cần text encoder.

- Prompt embedding tính trước (scripts/cache-prompt-embeds.py) → pipeline dựng với text_encoder=None.
- Mỗi prompt sinh G ảnh một lượt (num_images_per_prompt), lấy latent đã pack (đích x0 cho loss NFT)
  rồi giải mã VAE ra ảnh RGB để OCR chấm reward.
"""
import torch
from PIL import Image


def build_rollout_pipeline(model_dir: str, transformer):
    from diffusers import QwenImage21Pipeline
    pipe = QwenImage21Pipeline.from_pretrained(model_dir, transformer=transformer, text_encoder=None,
                                               torch_dtype=torch.bfloat16)
    pipe.vae.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _rgb_on_white(img: Image.Image) -> Image.Image:
    if img.mode != "RGBA":
        return img.convert("RGB")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    return bg


@torch.no_grad()
def decode_packed_latents(pipe, latents: torch.Tensor, res: int) -> list:
    """latents (G, N, 64) đã chuẩn hoá → list ảnh PIL RGB."""
    vae = pipe.vae
    z = vae.config.z_dim
    lat = pipe._unpack_latents(latents, res, res, pipe.vae_scale_factor).to(vae.dtype)
    mean = torch.tensor(vae.config.latents_mean).view(1, z, 1, 1, 1).to(lat.device, lat.dtype)
    std = torch.tensor(vae.config.latents_std).view(1, z, 1, 1, 1).to(lat.device, lat.dtype)
    images = vae.decode(lat * std + mean, return_dict=False)[0][:, :, 0]
    return [_rgb_on_white(im) for im in pipe.image_processor.postprocess(images, output_type="pil")]


@torch.no_grad()
def rollout_group(pipe, prompt_embeds: torch.Tensor, group_size: int, res: int, steps: int, seeds: list):
    """prompt_embeds (L, D) → (latents (G, N, 64) bf16, ảnh PIL)."""
    gens = [torch.Generator("cuda").manual_seed(s) for s in seeds]
    out = pipe(prompt_embeds=prompt_embeds[None].to("cuda", torch.bfloat16), num_images_per_prompt=group_size,
               height=res, width=res, num_inference_steps=steps, generator=gens, output_type="latent")
    latents = out.images
    return latents, decode_packed_latents(pipe, latents, res)
