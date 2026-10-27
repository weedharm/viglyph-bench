"""Tính trước latent VAE, text embedding và mask token cho tập ảnh tổng hợp (GPU, bf16).

Mỗi mẫu một file <cache>/<id>.pt: {"latents": (4096, 64) bf16 đã chuẩn hoá mean/std và pack,
"prompt_embeds": (L, 4096) bf16 (qua pipe.encode_prompt, có hook bỏ RMSNorm cuối), "token_mask": (4096,) uint8}.
Ghi thêm <cache>/index.jsonl. Resume: bỏ qua mẫu đã có file.

Usage: python scripts/cache-latents-and-prompt-embeds.py --model models/Qwen-Image-2.1 \
           --data-root data --metadata data/synth/metadata.jsonl --cache data/cache/synth_v1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vitext.train.flow_matching_text_mask_loss import token_mask_from_pixel_mask  # noqa: E402


class SynthImages(torch.utils.data.Dataset):
    def __init__(self, rows, data_root):
        self.rows, self.root = rows, Path(data_root)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = np.asarray(Image.open(self.root / r["image"]).convert("RGBA"), dtype=np.float32) / 127.5 - 1.0
        if r.get("mask"):
            mask = (np.asarray(Image.open(self.root / r["mask"]).convert("L")) > 127).astype(np.float32)
        else:  # ảnh tự sinh (RFT) không có mask chữ -> mask rỗng, train với λ=0
            mask = np.zeros(img.shape[:2], dtype=np.float32)
        return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(mask)[None], i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--metadata", default=str(ROOT / "data/synth/metadata.jsonl"))
    ap.add_argument("--cache", default=str(ROOT / "data/cache/synth_v1"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from diffusers import QwenImage21Pipeline
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(ln) for ln in open(args.metadata, encoding="utf-8")][: args.limit]
    todo = [r for r in rows if not (cache / f"{r['id']}.pt").exists()]
    print(f"{len(rows)} samples, {len(todo)} to cache", flush=True)

    t0 = time.time()
    pipe = QwenImage21Pipeline.from_pretrained(args.model, transformer=None, torch_dtype=torch.bfloat16).to("cuda")
    vae = pipe.vae.eval()
    z = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean).view(1, z, 1, 1, 1).to("cuda", torch.bfloat16)
    std = torch.tensor(vae.config.latents_std).view(1, z, 1, 1, 1).to("cuda", torch.bfloat16)
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)

    loader = torch.utils.data.DataLoader(SynthImages(todo, args.data_root), batch_size=args.batch,
                                         num_workers=args.workers, shuffle=False)
    index_f = open(cache / "index.jsonl", "a", encoding="utf-8")
    n_done, t1 = 0, time.time()
    with torch.no_grad():
        for imgs, masks, idx in loader:
            batch_rows = [todo[i] for i in idx.tolist()]
            x = imgs.to("cuda", torch.bfloat16)[:, :, None]  # (B, 4, 1, H, W), RGBA trong [-1, 1]
            lat = (vae.encode(x).latent_dist.mode() - mean) / std  # (B, 64, 1, h, w)
            b, c, _, h, w = lat.shape
            lat = lat.view(b, c, h * w).transpose(1, 2).contiguous()  # pack -> (B, h·w, 64)
            embeds, emb_mask, _ = pipe.encode_prompt([r["caption"] for r in batch_rows], device="cuda")
            tok_mask = token_mask_from_pixel_mask(masks).to(torch.uint8)
            assert tok_mask.shape[1] == h * w, (tok_mask.shape, h, w)
            for j, r in enumerate(batch_rows):
                n_tok = int(emb_mask[j].sum()) if emb_mask is not None else embeds.shape[1]
                torch.save({"latents": lat[j].cpu(), "prompt_embeds": embeds[j, :n_tok].cpu(),
                            "token_mask": tok_mask[j].cpu(), "caption": r["caption"]}, cache / f"{r['id']}.pt")
                index_f.write(json.dumps({"id": r["id"], "n_text_tokens": n_tok, "latent_hw": [h, w],
                                          "text_frac": round(tok_mask[j].float().mean().item(), 4)}) + "\n")
            index_f.flush()
            n_done += len(batch_rows)
            if n_done % (args.batch * 50) == 0 or n_done == len(todo):
                rate = (time.time() - t1) / n_done
                print(f"{n_done}/{len(todo)} {rate:.2f}s/img peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB",
                      flush=True)
    print("CACHE_DONE", flush=True)


if __name__ == "__main__":
    main()
