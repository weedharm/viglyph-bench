"""Test "lỗi cài sẵn" cho OCR: render chữ cố tình sai dấu, xem OCR chép trung thực hay tự sửa.

Mỗi chuỗi gốc sinh 1 biến thể lỗi (đổi dấu thanh / bỏ dấu thanh / bỏ dấu phụ ở một âm tiết),
cộng một nhóm đối chứng không lỗi. Ghi <out>/meta.jsonl + <out>/images/*.png.
Usage: python build-diacritic-injection-ocr-test.py --texts bench/viglyph/draft-v0-texts.txt --out outputs/ocr_injection
"""
import argparse
import json
import random
import textwrap
import unicodedata
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TONES = ["́", "̀", "̉", "̃", "̣"]
DIACS = ["̂", "̛", "̆"]
FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"]


def corrupt(text: str, rng: random.Random):
    """Sửa đúng một ký tự có dấu; trả về (chuỗi lỗi, loại lỗi) hoặc None."""
    chars = list(text)
    cands = []
    for i, ch in enumerate(chars):
        nfd = unicodedata.normalize("NFD", ch)
        has_tone = any(m in TONES for m in nfd[1:])
        has_diac = any(m in DIACS for m in nfd[1:]) or ch in "đĐ"
        if has_tone:
            cands += [(i, "tone_swap"), (i, "tone_drop")]
        if has_diac:
            cands.append((i, "diac_drop"))
    if not cands:
        return None
    i, kind = rng.choice(cands)
    nfd = unicodedata.normalize("NFD", chars[i])
    if kind == "tone_swap":
        old = next(m for m in nfd if m in TONES)
        nfd = nfd.replace(old, rng.choice([t for t in TONES if t != old]))
    elif kind == "tone_drop":
        nfd = "".join(m for m in nfd if m not in TONES)
    else:
        nfd = {"đ": "d", "Đ": "D"}.get(chars[i]) or "".join(m for m in nfd if m not in DIACS)
    # Dấu mũ/trăng và dấu thanh cùng lớp kết hợp 230: phải đặt dấu phụ trước dấu thanh thì NFC mới gộp được.
    marks = sorted(nfd[1:], key=lambda m: (m in TONES, m))
    chars[i] = unicodedata.normalize("NFC", nfd[0] + "".join(marks))
    return "".join(chars), kind


def render(text: str, font_path: str, rng: random.Random, size: int = 72, width: int = 1024) -> Image.Image:
    """Nền màu nhạt + nhiễu nhẹ: Qwen3-VL-8B đọc hỏng ảnh nền trắng tuyệt đối (đã quan sát)."""
    font = ImageFont.truetype(font_path, size)
    lines = textwrap.wrap(text, width=max(8, int(width / (size * 0.62))))
    line_h = int(size * 1.5)
    height = max(512, line_h * len(lines) + 160)
    bg = tuple(rng.randint(200, 250) for _ in range(3))
    noise = Image.effect_noise((width, height), 12).convert("RGB")
    img = Image.blend(Image.new("RGB", (width, height), bg), noise, 0.08)
    draw = ImageDraw.Draw(img)
    y0 = (height - line_h * len(lines)) / 2
    for k, line in enumerate(lines):
        w = draw.textlength(line, font=font)
        draw.text(((width - w) / 2, y0 + k * line_h), line, font=font, fill=(25, 25, 35))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-control", type=int, default=20)
    args = ap.parse_args()
    rng = random.Random(7)
    texts = [ln.strip() for ln in open(args.texts, encoding="utf-8") if ln.strip() and not ln.startswith("#")]
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, text in enumerate(texts):
        res = corrupt(text, rng)
        if res is None:
            continue
        rows.append({"id": f"inj_{idx:04d}", "original": text, "shown": res[0], "kind": res[1]})
    for idx in rng.sample(range(len(texts)), args.n_control):
        rows.append({"id": f"ctl_{idx:04d}", "original": texts[idx], "shown": texts[idx], "kind": "control"})
    with open(out / "meta.jsonl", "w", encoding="utf-8") as f:
        for k, r in enumerate(rows):
            r["font"] = Path(FONTS[k % len(FONTS)]).stem
            render(r["shown"], FONTS[k % len(FONTS)], rng).save(out / "images" / f"{r['id']}.png")
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"rendered {len(rows)} images -> {out}")


if __name__ == "__main__":
    main()
