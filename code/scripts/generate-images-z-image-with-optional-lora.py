"""Sinh ảnh ViGlyph-Bench bằng Z-Image (diffusers ZImagePipeline), có thể kèm LoRA Z-LAN (PR Letters 2026) để so sánh
với một công trình đã công bố. Thiết lập mặc định theo README của Z-LAN: 50 bước, guidance 7,5.

- LoRA Z-LAN lưu khoá dạng PEFT "<module>.lora_A.default.weight" theo tên module của transformer diffusers: nạp bằng
  PEFT (add_adapter + set_peft_model_state_dict) và dừng ngay nếu có khoá không khớp. alpha = rank (tỉ lệ 1) vì file
  không ghi alpha.
- Resume: bỏ qua ảnh đã có. Ghi <out>/gen_log.jsonl như generate-images-qwen-image-21.py.
Usage: python scripts/generate-images-z-image-with-optional-lora.py --model models/Z-Image \
           --prompts bench/viglyph/prompts_v1_en_zlan.jsonl --out outputs/v1_zimage_zlan --lora models/zlan/zlan-Vietnamese_20.safetensors
"""
import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import ZImagePipeline


def load_peft_lora(transformer, path: str, adapter: str = "zlan") -> int:
    from peft import LoraConfig, set_peft_model_state_dict
    from safetensors.torch import load_file
    sd = load_file(path)
    sd = {k.replace(".default.", "."): v for k, v in sd.items()}
    targets = sorted({k.split(".lora_")[0] for k in sd})
    rank = next(v.shape[0] for k, v in sd.items() if ".lora_A." in k)
    transformer.add_adapter(LoraConfig(r=rank, lora_alpha=rank, target_modules=targets, init_lora_weights=False),
                            adapter_name=adapter)
    res = set_peft_model_state_dict(transformer, sd, adapter_name=adapter)
    unexpected = list(getattr(res, "unexpected_keys", []) or [])
    if unexpected:
        raise RuntimeError(f"LoRA có {len(unexpected)} khoá không khớp, ví dụ {unexpected[:3]}")
    n_loaded = sum(1 for n, _ in transformer.named_parameters() if f".{adapter}." in n and "lora_" in n)
    if n_loaded != len(sd):
        raise RuntimeError(f"LoRA nạp {n_loaded} tham số, file có {len(sd)}")
    transformer.set_adapter(adapter)
    return rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--lora", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = [json.loads(ln) for ln in open(args.prompts, encoding="utf-8")]
    t0 = time.time()
    pipe = ZImagePipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    if args.lora:
        rank = load_peft_lora(pipe.transformer, args.lora)
        print(f"LoRA {args.lora} nạp đủ khoá (rank {rank})", flush=True)
    print(f"loaded in {time.time() - t0:.1f}s; {len(rows)} prompts @ {args.res}px, {args.steps} steps, cfg {args.guidance}", flush=True)
    log_f = open(out / "gen_log.jsonl", "a", encoding="utf-8")
    for k, r in enumerate(rows):
        img_path = out / "images" / f"{r['id']}_s{r['seed']}.png"
        if img_path.exists():
            continue
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        image = pipe(prompt=r["prompt"], width=args.res, height=args.res, num_inference_steps=args.steps,
                     guidance_scale=args.guidance, generator=torch.Generator("cuda").manual_seed(r["seed"])).images[0]
        dt = time.time() - t1
        image.convert("RGB").save(img_path)
        log_f.write(json.dumps({"id": r["id"], "seed": r["seed"], "res": args.res, "steps": args.steps,
                                "guidance": args.guidance, "lora": args.lora, "sec": round(dt, 2),
                                "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}) + "\n")
        log_f.flush()
        print(f"[{k + 1}/{len(rows)}] {r['id']} {dt:.1f}s", flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
