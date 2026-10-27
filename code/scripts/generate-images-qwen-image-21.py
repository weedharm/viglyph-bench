"""Sinh ảnh ViGlyph-Bench bằng Qwen-Image-2.1 (diffusers), không prompt rewriter, seed cố định.

- Resume: bỏ qua ảnh đã có.
- Thứ tự xen kẽ giữa các bucket để nếu dừng sớm thì kết quả vẫn cân bằng.
- Ghi giây/ảnh và VRAM đỉnh từng ảnh vào <out>/gen_log.jsonl.
- --lora: nạp LoRA (thư mục do train-lora-qwen-image-21.py lưu) và hợp nhất vào trọng số với --lora-scale.
- --pipeline qwen-image: Qwen-Image bản đầu (2025, CFG thật) để so sánh; dùng --true-cfg và --negative-prompt.
Usage:
  python generate-images-qwen-image-21.py --prompts p.jsonl --out outputs/x --res 1024 [--per-bucket 12]
  python generate-images-qwen-image-21.py --prompts p.jsonl --out outputs/x --lora outputs/lora_l0/final
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from diffusers import QwenImage21Pipeline, QwenImagePipeline
from PIL import Image


def interleave_by_bucket(rows, per_bucket=None):
    groups = defaultdict(list)
    for r in rows:
        groups[r["bucket"]].append(r)
    if per_bucket:  # chọn đều K mẫu mỗi bucket
        for b, items in groups.items():
            step = max(len(items) / per_bucket, 1)
            groups[b] = [items[int(i * step)] for i in range(min(per_bucket, len(items)))]
    keyed = []
    for items in groups.values():
        keyed += [((i + 0.5) / len(items), r["id"], r) for i, r in enumerate(items)]
    return [r for _, _, r in sorted(keyed, key=lambda x: (x[0], x[1]))]


def to_rgb_on_white(img: Image.Image) -> Image.Image:
    if img.mode != "RGBA":
        return img.convert("RGB")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    return bg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/models/Qwen-Image-2.1")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--per-bucket", type=int, default=None)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--pipeline", choices=["qwen-image-2.1", "qwen-image"], default="qwen-image-2.1")
    ap.add_argument("--true-cfg", type=float, default=4.0, help="chỉ cho qwen-image (bản đầu, chưa chưng cất CFG)")
    ap.add_argument("--negative-prompt", default=" ")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in open(args.prompts, encoding="utf-8")]
    rows = interleave_by_bucket(rows, args.per_bucket)

    t0 = time.time()
    pipe_cls = QwenImagePipeline if args.pipeline == "qwen-image" else QwenImage21Pipeline
    pipe = pipe_cls.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    if args.lora:
        pipe.load_lora_weights(args.lora, adapter_name="vitext")
        pipe.fuse_lora(lora_scale=args.lora_scale, components=["transformer"])
        print(f"LoRA {args.lora} fused (scale {args.lora_scale})", flush=True)
    print(f"loaded in {time.time() - t0:.1f}s; {len(rows)} prompts @ {args.res}px", flush=True)

    log_f = open(out / "gen_log.jsonl", "a", encoding="utf-8")
    for k, r in enumerate(rows):
        seed = r["seed"] + args.seed_offset
        img_path = out / "images" / f"{r['id']}_s{seed}.png"
        if img_path.exists():
            continue
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t1 = time.time()
        extra = {"negative_prompt": args.negative_prompt, "true_cfg_scale": args.true_cfg} if args.pipeline == "qwen-image" else {}
        image = pipe(
            prompt=r["prompt"], width=args.res, height=args.res, num_inference_steps=args.steps,
            generator=torch.Generator("cuda").manual_seed(seed), **extra,
        ).images[0]
        torch.cuda.synchronize()
        dt = time.time() - t1
        to_rgb_on_white(image).save(img_path)
        rec = {
            "id": r["id"], "seed": seed, "res": args.res, "steps": args.steps, "image": img_path.name,
            "sec": round(dt, 2), "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),  # GiB (log cũ ghi nhầm tên peak_vram_gb)
            "mode": image.mode, "pipeline": args.pipeline, "lora": args.lora, "lora_scale": args.lora_scale if args.lora else None,
        }
        log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log_f.flush()
        print(f"[{k + 1}/{len(rows)}] {r['id']} {dt:.1f}s {rec['peak_vram_gib']}GiB", flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
