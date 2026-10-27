"""Thành phần vẽ ảnh chữ tổng hợp loại A bằng PIL: bảng màu có tên tiếng Việt, kho font, xếp chữ vừa khung, mask chữ.

Mask là nét mực của chữ (kể cả viền), vẽ cùng tham số với ảnh; loss sẽ max-pool mask về lưới token 16 px.
Chiều cao dòng đo theo nét thật của chuỗi mẫu có dấu chồng trên chữ hoa ("ẪỖỰẶ") để dấu không chạm dòng trên.
"""
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

PALETTE = {
    "trắng": (246, 246, 241), "kem": (244, 233, 205), "be": (226, 211, 181), "đen": (24, 24, 27),
    "đỏ": (200, 32, 42), "đỏ đô": (122, 22, 34), "cam": (238, 118, 32), "vàng": (247, 202, 42),
    "vàng nhạt": (250, 236, 170), "xanh lá": (36, 135, 70), "xanh rêu": (68, 88, 52), "xanh dương": (32, 92, 192),
    "xanh navy": (22, 36, 82), "xanh ngọc": (28, 158, 158), "xanh pastel": (182, 216, 236),
    "hồng pastel": (246, 194, 204), "hồng đậm": (206, 46, 110), "tím": (108, 52, 140), "nâu": (108, 70, 42),
    "xám": (122, 122, 128), "xám nhạt": (214, 214, 218),
}
SERIF = {"Lora", "Merriweather", "NotoSerif", "PlayfairDisplay"}
HAND = {"DancingScript", "Pacifico", "PatrickHand"}
WEIGHTS = ["Regular", "Medium", "SemiBold", "Bold", "ExtraBold", "Black"]
_REF_TALL = "ẪỖỰẶẲÉ Ậgjpyq"  # chuỗi đo chiều cao dòng: dấu chồng trên chữ hoa + chân chữ


def luminance(rgb) -> float:
    c = [v / 255 for v in rgb]
    c = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def contrast(a, b) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def pick_color(rng, against, min_contrast=4.0, exclude=()):
    """Chọn ngẫu nhiên một màu (tên, rgb) đủ tương phản với màu nền `against`."""
    names = [n for n, c in PALETTE.items() if n not in exclude and contrast(c, against) >= min_contrast]
    name = rng.choice(names or ["đen" if luminance(against) > 0.2 else "trắng"])
    return name, PALETTE[name]


@dataclass
class FontChoice:
    name: str        # tên file gốc, ví dụ "BeVietnamPro", "Inter"
    path: str
    weight: str      # Regular..Black
    category: str    # sans | serif | hand
    instances: list = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.name}-{self.weight}"


class FontCatalog:
    """Đọc fonts_ok.json, gom theo họ; font tĩnh lấy độ đậm từ tên file."""

    def __init__(self, fonts_json: Path):
        fonts_dir = Path(fonts_json).parent
        self.families = {}
        for e in json.loads(Path(fonts_json).read_text(encoding="utf-8")):
            fname = Path(e["path"]).name
            stem = fname.split("%5B")[0].split("-")[0]
            cat = "serif" if stem in SERIF else "hand" if stem in HAND else "sans"
            path = str(fonts_dir / fname)
            if e["variable"]:
                for w in WEIGHTS:
                    if w in e["instances"]:
                        self.families.setdefault(stem, []).append(FontChoice(stem, path, w, cat, e["instances"]))
            else:
                w = fname.rsplit("-", 1)[-1].removesuffix(".ttf")
                self.families.setdefault(stem, []).append(FontChoice(stem, path, w if w in WEIGHTS else "Regular", cat))

    def sample(self, rng, category_weights: dict, min_weight: str = "Regular", exclude=()) -> FontChoice:
        cats = list(category_weights)
        cat = rng.choices(cats, weights=[category_weights[c] for c in cats])[0]
        fams = [f for f, v in self.families.items() if v[0].category == cat and f not in exclude]
        opts = [c for c in self.families[rng.choice(fams)] if WEIGHTS.index(c.weight) >= WEIGHTS.index(min_weight)]
        return rng.choice(opts or self.families[fams[0]])


@lru_cache(maxsize=512)
def load_font(path: str, size: int, weight: str, variable: bool) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(path, size)
    if variable:
        font.set_variation_by_name(weight)
    return font


def get_font(fc: FontChoice, size: int) -> ImageFont.FreeTypeFont:
    return load_font(fc.path, size, fc.weight, bool(fc.instances))


def line_metrics(font) -> tuple:
    """(độ cao phía trên baseline, phía dưới baseline) theo nét thật của chuỗi có dấu chồng."""
    x0, top, x1, bottom = font.getbbox(_REF_TALL, anchor="ls")
    return -top, bottom


def wrap_words(text: str, font, max_w: float) -> list:
    lines, cur = [], ""
    for w in text.split():
        trial = f"{cur} {w}".strip()
        if cur and font.getlength(trial) > max_w:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    return lines + [cur] if cur else lines


def fit_text(text, fc: FontChoice, box_w, box_h, max_size, min_size, max_lines=6, spacing=1.12):
    """Cỡ chữ lớn nhất (tìm nhị phân) để chuỗi xuống dòng vừa khung. Trả về (font, lines, line_h) hoặc None."""
    def layout(size):
        font = get_font(fc, size)
        lines = wrap_words(text, font, box_w)
        up, down = line_metrics(font)
        line_h = (up + down) * spacing
        ok = (len(lines) <= max_lines and line_h * len(lines) <= box_h
              and all(font.getlength(ln) <= box_w for ln in lines))
        return ok, (font, lines, line_h)
    lo, hi, best = min_size, max_size, None
    while lo <= hi:
        mid = (lo + hi) // 2
        ok, res = layout(mid)
        if ok:
            best, lo = res, mid + 1
        else:
            hi = mid - 1
    return best


def draw_text_block(img, mask, fitted, box, fill, align="center", valign="center", stroke=0,
                    stroke_fill=None, shadow=None, chalk_rng=None):
    """Vẽ khối chữ đã fit vào box=(x0,y0,x1,y1) trên ảnh RGB và mask L. Trả về bbox nét mực trong mask."""
    font, lines, line_h = fitted
    up, _ = line_metrics(font)
    x0, y0, x1, y1 = box
    total_h = line_h * len(lines)
    top = y0 + {"top": 0, "center": (y1 - y0 - total_h) / 2, "bottom": y1 - y0 - total_h}[valign]
    layer = Image.new("L", img.size, 0)  # nét mực của riêng khối này
    ld = ImageDraw.Draw(layer)
    for i, ln in enumerate(lines):
        w = font.getlength(ln)
        x = {"left": x0, "center": (x0 + x1 - w) / 2, "right": x1 - w}[align]
        ld.text((x, top + i * line_h + up), ln, font=font, fill=255, anchor="ls", stroke_width=stroke)
    if shadow:
        dx, dy, color = shadow
        sh = layer.filter(ImageFilter.GaussianBlur(3)).point(lambda v: int(v * 0.55))
        img.paste(Image.new("RGB", img.size, color), (dx, dy), sh)
    if stroke and stroke_fill is not None:
        img.paste(Image.new("RGB", img.size, stroke_fill), (0, 0), layer)
        inner = Image.new("L", img.size, 0)
        idraw = ImageDraw.Draw(inner)
        for i, ln in enumerate(lines):
            w = font.getlength(ln)
            x = {"left": x0, "center": (x0 + x1 - w) / 2, "right": x1 - w}[align]
            idraw.text((x, top + i * line_h + up), ln, font=font, fill=255, anchor="ls")
        paint = inner
    else:
        paint = layer
    if chalk_rng is not None:  # phấn: nét lốm đốm
        noise = chalk_rng.integers(150, 256, size=(img.size[1], img.size[0]), dtype=np.uint8)
        paint = Image.fromarray((np.asarray(paint, dtype=np.uint16) * noise // 255).astype(np.uint8))
    img.paste(Image.new("RGB", img.size, fill), (0, 0), paint)
    mask.paste(255, (0, 0), layer)
    return layer.getbbox()


def vertical_gradient(size, c1, c2) -> Image.Image:
    w, h = size
    t = np.linspace(0, 1, h)[:, None, None]
    arr = (np.array(c1)[None, None] * (1 - t) + np.array(c2)[None, None] * t).repeat(w, axis=1)
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def add_noise(img, rng: np.random.Generator, amount=8) -> Image.Image:
    arr = np.asarray(img, dtype=np.int16) + rng.normal(0, amount, size=(img.size[1], img.size[0], 1)).astype(np.int16)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
