"""Tính trước text embedding (pipe.encode_prompt, có hook bỏ RMSNorm cuối) cho các file prompt jsonl.

Ghi một file .pt: {id: tensor (L, 4096) bf16}. Dùng cho RL để khỏi giữ text encoder 16 GB trên GPU khi rollout.
Usage: python scripts/cache-prompt-embeds.py --model models/Qwen-Image-2.1 \
           --prompts bench/selftrain/prompts_rl_v1.jsonl bench/viglyph/prompts_dev_v1.jsonl --out data/cache/rl_prompt_embeds_v1.pt
"""
import argparse
import json
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    from diffusers import QwenImage21Pipeline
    rows = [json.loads(ln) for p in args.prompts for ln in open(p, encoding="utf-8")]
    t0 = time.time()
    pipe = QwenImage21Pipeline.from_pretrained(args.model, transformer=None, torch_dtype=torch.bfloat16).to("cuda")
    out = {}
    with torch.no_grad():
        for i in range(0, len(rows), args.batch):
            chunk = rows[i:i + args.batch]
            emb, mask, _ = pipe.encode_prompt([r["prompt"] for r in chunk], device="cuda")
            for j, r in enumerate(chunk):
                n = int(mask[j].sum()) if mask is not None else emb.shape[1]
                out[r["id"]] = emb[j, :n].cpu()
    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"{len(out)} embeddings -> {args.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
