"""Chạy Qwen-Image-2.1 ở chế độ sửa ảnh trên benchmark MultiTextEdit (lwcheng/MultiTextEdit, Apache-2.0).

Mỗi mẫu: ảnh gốc 1.jpg (đã có chữ của ngôn ngữ đó) + lệnh tiếng Anh, chữ đích đặt trong nháy đơn.
Ghi <out>/images/<id>.png, <out>/gen_log.jsonl và <out>/manifest.jsonl (chữ đích theo từng loại thao tác).
Seed cố định theo thứ tự mẫu; resume: bỏ qua ảnh đã có. Ảnh ra theo tỉ lệ ảnh gốc, diện tích khoảng 1024².

Usage: python scripts/edit-images-multitextedit-qwen-image-21.py --model models/Qwen-Image-2.1 \
           --data data/external/MultiTextEdit/unzipped/final_data_1024version --lang vi --out outputs/mte_base_vi
"""
import argparse
import json
import re
import time
from pathlib import Path

BASE_SEED = 20261008


_QUOTE_PAIRS = [("\u201c", "\u201d"), ("\u2018", "\u2019"), ('"', '"')]  # “…”, ‘…’, "…"; '…' chỉ dùng khi không có loại khác


def quoted_texts(prompt: str) -> list:
    """Chuỗi trong ngoặc theo thứ tự xuất hiện; lệnh của MultiTextEdit dùng lẫn 4 kiểu ngoặc."""
    found = []
    for o, c in _QUOTE_PAIRS:
        found += [(m.start(), m.group(1)) for m in re.finditer(re.escape(o) + r"(.+?)" + re.escape(c), prompt)]
    if not found:  # nháy đơn: tránh cắt ở dấu nháy trong từ (Mother's)
        found = [(m.start(), m.group(1)) for m in re.finditer(r"(?<!\w)'(.+?)'(?!\w)", prompt)]
    return [t.strip() for _, t in sorted(found)]


def targets_of(row: dict) -> dict:
    """Chữ đích cần có trong ảnh ra (expected), chữ phải biến mất (removed), chữ neo vị trí (anchor)."""
    quoted = quoted_texts(row["prompt"])
    m = row["editing_method"]
    if not quoted:  # lệnh cho cả ảnh ("Straighten the tilted text..."): vẫn sinh ảnh nhưng không có chữ đích để chấm
        return {"expected": None, "removed": None, "anchor": None}
    if m == "text_delete":
        return {"expected": None, "removed": quoted[0], "anchor": None}
    if m == "exchange":
        return {"expected": quoted[1], "removed": quoted[0], "anchor": None}
    if m == "insert":
        return {"expected": quoted[0], "removed": None, "anchor": quoted[1] if len(quoted) > 1 else None}
    # color, relocation (kể cả "Swap the positions of A and B"), scaling, change_font
    return {"expected": quoted[0], "removed": None, "anchor": quoted[1] if len(quoted) > 1 else None}


def load_rows(data_root: Path, lang: str) -> list:
    rows = []
    for f in sorted(data_root.glob(f"*/*/{lang}/*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        d = f.parent
        rows.append({**r, **targets_of(r), "input_path": str(d / r["input_image"]), "ref_path": str(d / r["output_image"])})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--lang", default="vi")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--lora", default=None)
    args = ap.parse_args()

    import torch
    from diffusers import QwenImage21Pipeline
    from PIL import Image

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path(args.data), args.lang)[: args.limit]
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    (out / "refs").mkdir(exist_ok=True)  # ảnh đích do người sửa, đặt tên <id>.png để OCR làm trần so sánh
    for r in rows:
        link = out / "refs" / f"{r['id']}.png"
        if not link.exists():
            link.symlink_to(Path(r["ref_path"]).resolve())
    print(f"{len(rows)} samples ({args.lang})", flush=True)

    t0 = time.time()
    pipe = QwenImage21Pipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    if args.lora:
        pipe.load_lora_weights(args.lora, adapter_name="vitext")
        pipe.fuse_lora(components=["transformer"])
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)

    log_f = open(out / "gen_log.jsonl", "a", encoding="utf-8")
    for k, r in enumerate(rows):
        img_path = out / "images" / f"{r['id']}.png"
        if img_path.exists():
            continue
        seed = BASE_SEED + k
        src = Image.open(r["input_path"]).convert("RGB")
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        res = pipe(prompt=r["prompt"], image=src, num_inference_steps=args.steps,
                   generator=torch.Generator("cuda").manual_seed(seed)).images[0]
        dt = time.time() - t1
        res.convert("RGB").save(img_path)
        rec = {"id": r["id"], "seed": seed, "sec": round(dt, 2), "in_size": src.size, "out_size": res.size,
               "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2), "lora": args.lora}
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        print(f"[{k + 1}/{len(rows)}] {r['id']} {r['editing_method']} {dt:.1f}s {rec['peak_vram_gib']}GiB", flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
