"""EasyOCR (CRNN + CTC, 'vi') làm OCR "trung thực": gần như không có mô hình ngôn ngữ nên không tự sửa dấu.

Test cài lỗi (E1): EasyOCR tự sửa 0% lỗi cài sẵn (Vintern 43%), nhưng đọc nhiễu hơn với font cách điệu.
Dùng làm OCR chấm thứ ba (cận dưới) và làm thành phần reward khó bị "lừa" bằng dấu mơ hồ.
"""
import numpy as np


def reading_order_text(results) -> str:
    """results của readtext: [(box 4 điểm, text, conf)] → các dòng nối bằng '\n' theo thứ tự đọc.

    Gom khung cùng dòng khi tâm y nằm trong nửa chiều cao dòng trước; trong dòng xếp theo x.
    """
    items = []
    for box, text, _ in results:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append({"y0": min(ys), "y1": max(ys), "x": min(xs), "text": text})
    items.sort(key=lambda r: (r["y0"] + r["y1"]) / 2)
    lines = []
    for it in items:
        cy = (it["y0"] + it["y1"]) / 2
        if lines and abs(cy - lines[-1]["cy"]) < 0.5 * max(1, lines[-1]["h"]):
            lines[-1]["items"].append(it)
        else:
            lines.append({"cy": cy, "h": it["y1"] - it["y0"], "items": [it]})
    return "\n".join(" ".join(i["text"] for i in sorted(ln["items"], key=lambda r: r["x"])) for ln in lines)


class EasyOCRReader:
    """read_batch(list[PIL.Image]) → list[str], cùng giao diện với Qwen3VLOCR.read_batch."""

    engine = "easyocr-vi"

    def __init__(self, langs=("vi",), gpu=True):
        import easyocr
        self.reader = easyocr.Reader(list(langs), gpu=gpu, download_enabled=False)

    def read_batch(self, images, **_):
        return [reading_order_text(self.reader.readtext(np.asarray(im.convert("RGB")))) for im in images]
