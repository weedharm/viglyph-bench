"""Kiểm tra nhanh Qwen3-VL định vị khung chữ (dùng cho loss vùng chữ trong RL): vẽ khung lên ảnh và lưu lưới.

Usage: python scripts/check-qwen3vl-text-grounding.py --images "outputs/rl_nft_v1/samples/*.jpg" --out outputs/grounding_check.png
"""
import argparse
import glob
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vitext.ocr.qwen3vl_ocr_wrapper import Qwen3VLOCR, locate_text_boxes_batch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=str(ROOT / "models/Qwen3-VL-8B-Instruct"))
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    files = sorted(glob.glob(args.images))[: args.n]
    ims = [Image.open(f).convert("RGB") for f in files]
    ocr = Qwen3VLOCR(args.model, args.model, "qwen3vl")
    boxes = locate_text_boxes_batch(ocr, ims)
    tile = 384
    sheet = Image.new("RGB", (4 * tile, ((len(ims) + 3) // 4) * tile), "white")
    for k, (im, bx) in enumerate(zip(ims, boxes)):
        t = im.resize((tile, tile))
        d = ImageDraw.Draw(t)
        for x1, y1, x2, y2 in bx:
            d.rectangle((x1 * tile / 1000, y1 * tile / 1000, x2 * tile / 1000, y2 * tile / 1000), outline=(255, 0, 0), width=3)
        sheet.paste(t, ((k % 4) * tile, (k // 4) * tile))
        print(Path(files[k]).name, len(bx), bx[:3])
    sheet.save(args.out)
    print(f"saved {args.out}; empty: {sum(not b for b in boxes)}/{len(boxes)}")


if __name__ == "__main__":
    main()
