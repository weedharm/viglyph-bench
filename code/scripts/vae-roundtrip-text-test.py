"""Kiểm tra VAE của Qwen-Image-2.1 có giữ dấu tiếng Việt không (Pha 1 plan.md).

Render chữ sạch ở nhiều cỡ (px chiều cao font) -> encode -> decode (bf16, như lúc suy luận).
Lưu orig/ và recon/ cùng tên để OCR so sánh; ghi PSNR toàn ảnh và dải ảnh crop để soát mắt.
Usage: python vae-roundtrip-text-test.py --out outputs/vae_roundtrip
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLQwenImage21
from PIL import Image, ImageDraw, ImageFont

TEXTS = [
    "Phở Bò Tái Nạm", "Đặc sản Mỹ Tho", "Ngẫm nghĩ rực rỡ", "Sữa chua nếp cẩm", "Ưu đãi lễ hội",
    "Tuổi trẻ ước mơ", "Lặng lẽ sững sờ", "Khuyến mãi đặc biệt", "Ẩm thực Huế", "Nghỉ ngơi bình yên",
    "MỖI NGÀY ĐỘC LẬP", "Giữ gìn tiếng Việt",
]
SIZES = [16, 24, 32, 48, 64, 96]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def render(text: str, size: int, w: int = 1024, h: int = 256) -> Image.Image:
    img = Image.new("RGB", (w, h), (238, 232, 220))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, size)
    tw = draw.textlength(text, font=font)
    draw.text(((w - tw) / 2, (h - size * 1.2) / 2), text, font=font, fill=(30, 30, 40))
    return img


def to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGBA")).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)[None, :, None]  # (1, 4, 1, H, W)


def to_image(x: torch.Tensor) -> Image.Image:
    arr = ((x[0, :3, 0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, "RGB")


def psnr(a: Image.Image, b: Image.Image) -> float:
    mse = np.mean((np.asarray(a, np.float32) - np.asarray(b, np.float32)) ** 2)
    return 99.0 if mse == 0 else 10 * math.log10(255 ** 2 / mse)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/models/Qwen-Image-2.1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    for sub in ("orig", "recon"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    vae = AutoencoderKLQwenImage21.from_pretrained(f"{args.model}/vae", torch_dtype=torch.bfloat16).to("cuda").eval()
    rows, strips = [], []
    for size in SIZES:
        for i, text in enumerate(TEXTS):
            name = f"vae_s{size:03d}_{i:02d}.png"
            orig = render(text, size)
            z = vae.encode(to_tensor(orig).to("cuda", torch.bfloat16)).latent_dist.mode()
            recon = to_image(vae.decode(z).sample)
            orig.save(out / "orig" / name)
            recon.save(out / "recon" / name)
            rows.append({"image": name, "text": text, "font_px": size, "psnr": round(psnr(orig, recon), 2)})
            if i < 2:  # dải soát mắt: crop vùng chữ, gốc trên / khôi phục dưới
                box = (256, 64, 768, 192)
                pair = Image.new("RGB", (512, 256), "white")
                pair.paste(orig.crop(box), (0, 0))
                pair.paste(recon.crop(box), (0, 128))
                strips.append(pair)
    with open(out / "meta.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sheet = Image.new("RGB", (512 * 2, 256 * len(SIZES)), "white")
    for k, s in enumerate(strips):
        sheet.paste(s, ((k % 2) * 512, (k // 2) * 256))
    sheet.save(out / "visual_check_orig_top_recon_bottom.png")
    for size in SIZES:
        ps = [r["psnr"] for r in rows if r["font_px"] == size]
        print(f"font {size}px: PSNR mean {sum(ps) / len(ps):.2f} dB")


if __name__ == "__main__":
    main()
