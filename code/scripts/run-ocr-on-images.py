"""Chạy OCR (vintern | qwen3vl) trên thư mục ảnh; ghi <out>/ocr_<engine>.jsonl (resume theo tên ảnh).

vintern cần venv /venv/ocr (transformers 4.46); qwen3vl cần venv chính (transformers >= 5.17).
Usage: python run-ocr-on-images.py --engine vintern --images outputs/x/images --out outputs/x [--watch]
--watch: tiếp tục quét ảnh mới cho tới khi thấy file <images>/../GEN_DONE hoặc hết thời gian chờ.
--models-root: thư mục chứa model (mặc định $VITEXT_MODELS hoặc /workspace/models).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def load_engine(engine: str, models: str):
    if engine == "vintern":
        from vitext.ocr.vintern_ocr_wrapper import VinternOCR
        return VinternOCR(f"{models}/Vintern-1B-v3_5")
    from vitext.ocr.qwen3vl_ocr_wrapper import Qwen3VLOCR
    if engine == "qwen3vl-te":  # text_encoder của Qwen-Image-2.1 (đọc kém, chỉ để đối chiếu)
        root = f"{models}/Qwen-Image-2.1"
        return Qwen3VLOCR(f"{root}/text_encoder", f"{root}/processor", "qwen3vl-te-of-qwenimage21")
    path = f"{models}/Qwen3-VL-8B-Instruct"
    return Qwen3VLOCR(path, path, "qwen3vl-8b-instruct")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", choices=["vintern", "qwen3vl", "qwen3vl-te"], default="vintern")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--idle-timeout", type=int, default=600)
    ap.add_argument("--models-root", default=os.environ.get("VITEXT_MODELS", "/workspace/models"))
    args = ap.parse_args()

    img_dir, out_path = Path(args.images), Path(args.out) / f"ocr_{args.engine}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(line)["image"] for line in open(out_path, encoding="utf-8")}
    ocr = load_engine(args.engine, args.models_root)
    f = open(out_path, "a", encoding="utf-8")
    last_new = time.time()
    while True:
        todo = [p for p in sorted(img_dir.glob("*.png")) if p.name not in done]
        for p in todo:
            try:
                img = Image.open(p)
                img.load()
            except OSError:  # ảnh đang được ghi dở
                continue
            t0 = time.time()
            text = ocr.read(img)
            f.write(json.dumps({"image": p.name, "engine": ocr.name, "text": text,
                                "sec": round(time.time() - t0, 2)}, ensure_ascii=False) + "\n")
            f.flush()
            done.add(p.name)
            last_new = time.time()
            print(f"{p.name}: {text[:80]!r}", flush=True)
        if not args.watch or (img_dir.parent / "GEN_DONE").exists() and not todo:
            break
        if time.time() - last_new > args.idle_timeout:
            break
        time.sleep(10)
    print("OCR_DONE", flush=True)


if __name__ == "__main__":
    main()
