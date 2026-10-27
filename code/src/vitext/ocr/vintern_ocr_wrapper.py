"""Wrapper OCR Vintern-1B-v3.5 (5CD-AI, MIT). Chạy trong venv riêng (transformers 4.4x).

Prompt yêu cầu chép nguyên văn, không sửa chính tả, để hạn chế việc VLM "tự thêm dấu".
Giải mã greedy, không repetition_penalty (phạt lặp làm sai chuỗi kiểu "mắm tôm, mắm ruốc").
"""
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from .ocr_instruction_prompt import OCR_INSTRUCTION

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
OCR_QUESTION = "<image>\n" + OCR_INSTRUCTION


def _transform(size: int):
    return T.Compose([
        T.Lambda(lambda im: im.convert("RGB")),
        T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _dynamic_tiles(image: Image.Image, max_num: int = 6, size: int = 448):
    """Chia ảnh thành ô 448 theo tỉ lệ khung gần nhất + thumbnail (như model card)."""
    w, h = image.size
    ratios = sorted({(i, j) for n in range(1, max_num + 1) for i in range(1, n + 1)
                     for j in range(1, n + 1) if i * j <= max_num}, key=lambda x: x[0] * x[1])
    aspect = w / h
    best, best_diff = (1, 1), float("inf")
    for r in ratios:
        diff = abs(aspect - r[0] / r[1])
        if diff < best_diff or (diff == best_diff and w * h > 0.5 * size * size * r[0] * r[1]):
            best, best_diff = r, diff
    tw, th = size * best[0], size * best[1]
    resized = image.resize((tw, th))
    tiles = [resized.crop(((i % best[0]) * size, (i // best[0]) * size,
                           (i % best[0] + 1) * size, (i // best[0] + 1) * size))
             for i in range(best[0] * best[1])]
    if len(tiles) > 1:
        tiles.append(image.resize((size, size)))
    return tiles


class VinternOCR:
    name = "vintern-1b-v3.5"

    def __init__(self, model_path: str, max_tiles: int = 6, device: str = "cuda"):
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            trust_remote_code=True, use_flash_attn=False,
        ).eval().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.max_tiles, self.device, self.tf = max_tiles, device, _transform(448)

    @torch.inference_mode()
    def read(self, image: Image.Image, max_new_tokens: int = 256) -> str:
        tiles = _dynamic_tiles(image.convert("RGB"), self.max_tiles)
        pixel_values = torch.stack([self.tf(t) for t in tiles]).to(torch.bfloat16).to(self.device)
        gen_cfg = dict(max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, repetition_penalty=1.0)
        return self.model.chat(self.tokenizer, pixel_values, OCR_QUESTION, gen_cfg).strip()
