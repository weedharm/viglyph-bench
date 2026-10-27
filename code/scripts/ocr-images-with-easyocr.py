"""OCR ảnh bằng EasyOCR (CRNN + CTC, ngôn ngữ 'vi'): bộ nhận dạng ký tự gần như không có mô hình ngôn ngữ,
nên chép đúng nét đã vẽ hơn các OCR dạng VLM (Qwen3-VL, Vintern) vốn hay tự sửa dấu (xem E1).

Ghi <run>/ocr_easyocr.jsonl cùng định dạng các OCR khác ({"image", "engine", "text", "sec"}) để
score-baseline-from-ocr.py / score-ocr-injection-test.py chấm được ngay.

Usage: python scripts/ocr-images-with-easyocr.py --images outputs/baseline_1024/images --out outputs/baseline_1024
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vitext.ocr.easyocr_line_reader import reading_order_text  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", default="vi")
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()
    import easyocr
    reader = easyocr.Reader(args.langs.split(","), gpu=args.gpu)
    out = Path(args.out) / "ocr_easyocr.jsonl"
    done = {json.loads(ln)["image"] for ln in open(out, encoding="utf-8")} if out.exists() else set()
    files = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    with open(out, "a", encoding="utf-8") as f:
        for k, p in enumerate(files):
            if p.name in done:
                continue
            t0 = time.time()
            text = reading_order_text(reader.readtext(str(p)))
            f.write(json.dumps({"image": p.name, "engine": "easyocr-vi", "text": text,
                                "sec": round(time.time() - t0, 2)}, ensure_ascii=False) + "\n")
            f.flush()
            if k % 20 == 0:
                print(k, p.name, repr(text[:60]), flush=True)
    print("done →", out)


if __name__ == "__main__":
    main()
