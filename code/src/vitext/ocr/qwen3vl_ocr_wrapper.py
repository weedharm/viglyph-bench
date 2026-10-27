"""Wrapper OCR dùng Qwen3-VL-8B lấy từ text_encoder của Qwen-Image-2.1 (có lm_head).

Dùng để chấm/chấm chéo; trong RL (DiffusionNFT) đây là OCR reward, nên khi báo kết quả RL phải dùng thêm OCR khác
(Vintern) và nhãn người để chấm chéo (plan mục 6.6). Chạy trong venv chính
(transformers >= 5.17). Cùng câu lệnh "chép nguyên văn" như wrapper Vintern.
"""
import re

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .ocr_instruction_prompt import OCR_INSTRUCTION


class Qwen3VLOCR:
    """model_path/processor_path: Qwen3-VL-8B-Instruct gốc, hoặc text_encoder + processor của Qwen-Image-2.1.

    Lưu ý: text_encoder của 2.1 đọc đúng chỉ 45% ảnh đối chứng ở test cài lỗi -> không dùng làm OCR.
    """

    def __init__(self, model_path: str, processor_path: str, name: str, device: str = "cuda"):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, dtype=torch.bfloat16).eval().to(device)
        self.processor = AutoProcessor.from_pretrained(processor_path)
        self.device, self.name = device, name

    @torch.inference_mode()
    def read(self, image: Image.Image, max_new_tokens: int = 256) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image.convert("RGB")},
            {"type": "text", "text": OCR_INSTRUCTION},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

    @torch.inference_mode()
    def read_batch(self, images: list, max_new_tokens: int = 128) -> list:
        """Đọc nhiều ảnh một lượt (đệm trái), dùng khi chấm reward RL cho cả nhóm ảnh."""
        messages = [[{"role": "user", "content": [
            {"type": "image", "image": im.convert("RGB")},
            {"type": "text", "text": OCR_INSTRUCTION},
        ]}] for im in images]
        self.processor.tokenizer.padding_side = "left"
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True
        ).to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return [t.strip() for t in self.processor.batch_decode(new_tokens, skip_special_tokens=True)]


GROUNDING_INSTRUCTION = ('Locate every piece of text in the image. Output only a JSON list, one object per text line, '
                         'each with key "bbox_2d" as [x1, y1, x2, y2] in relative coordinates from 0 to 1000.')
_BOX = re.compile(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]")


def parse_relative_boxes(text: str) -> list:
    """Lấy các khung [x1, y1, x2, y2] (toạ độ tương đối 0–1000) từ câu trả lời; bỏ khung lỗi."""
    boxes = []
    for m in _BOX.finditer(text):
        x1, y1, x2, y2 = (float(v) for v in m.groups())
        if 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000:
            boxes.append((x1, y1, x2, y2))
    return boxes


def boxes_to_token_mask(boxes: list, grid: int, dilate: int = 1) -> torch.Tensor:
    """Khung tương đối 0–1000 → mask (grid·grid,) bool trên lưới token (hàng trước, như thứ tự pack)."""
    m = torch.zeros(grid, grid, dtype=torch.bool)
    for x1, y1, x2, y2 in boxes:
        c1, r1 = int(x1 / 1000 * grid) - dilate, int(y1 / 1000 * grid) - dilate
        c2, r2 = int(x2 / 1000 * grid) + 1 + dilate, int(y2 / 1000 * grid) + 1 + dilate
        m[max(0, r1):min(grid, r2), max(0, c1):min(grid, c2)] = True
    return m.flatten()


def locate_text_boxes_batch(ocr: "Qwen3VLOCR", images: list, max_new_tokens: int = 384) -> list:
    """Nhờ Qwen3-VL định vị các dòng chữ; trả về list khung cho từng ảnh (có thể rỗng)."""
    messages = [[{"role": "user", "content": [
        {"type": "image", "image": im.convert("RGB")},
        {"type": "text", "text": GROUNDING_INSTRUCTION},
    ]}] for im in images]
    ocr.processor.tokenizer.padding_side = "left"
    with torch.inference_mode():
        inputs = ocr.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True
        ).to(ocr.device)
        out = ocr.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    texts = ocr.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return [parse_relative_boxes(t) for t in texts]
