"""Huấn luyện LoRA cho DiT Qwen-Image-2.1 trên cache latent/embedding, loss flow matching có trọng số vùng chữ.

- LoRA rank/alpha 32 trên to_q/k/v, to_out.0 và 3 lớp MLP SwiGLU của cả 32 block (83,9 triệu tham số).
- DiT bf16 đông cứng, tham số LoRA fp32; gradient checkpointing; AdamW, warmup + cosine; clip grad 1.0.
- Loss: w = 1 + λ·m (λ = --lambda-mask; 0 là baseline), mục 6.2 plan.md.
- --bg-target base: token ngoài vùng chữ lấy đích là dự đoán của chính DiT gốc (tắt LoRA, không grad) thay vì
  ảnh tổng hợp → chỉ vùng chữ học từ dữ liệu, phần còn lại bị neo vào model gốc (chống trôi phong cách, bịa dòng).
- Ghi <out>/train_log.jsonl (loss, mse vùng chữ/nền, lr, grad norm, s/bước, VRAM đỉnh GiB).
- Lưu <out>/checkpoint-<step>/ (LoRA định dạng diffusers + training_state.pt để resume) và <out>/final/.
- --max-minutes: tự lưu và dừng trước giới hạn thời gian của job; chạy lại cùng lệnh để resume.

Usage: python scripts/train-lora-qwen-image-21.py --model models/Qwen-Image-2.1 --cache data/cache/synth_v1 \
           --out outputs/lora_l0 --steps 2000 --lambda-mask 0
"""
import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vitext.train.cached_latent_dataset import CachedLatentDataset, infinite_loader  # noqa: E402
from vitext.train.flow_matching_text_mask_loss import (  # noqa: E402
    mu_from_scheduler_config, noisy_input_and_target, sample_sigmas, weighted_flow_matching_loss)

LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "img_mlp.gate_layer", "img_mlp.proj", "img_mlp.out"]


def build_transformer(args):
    from diffusers import QwenImage21Transformer2DModel
    from peft import LoraConfig
    if args.tiny_test:  # DiT tí hon ngẫu nhiên để kiểm tra code trên GPU nhỏ
        tr = QwenImage21Transformer2DModel(num_layers=2, attention_head_dim=32, num_attention_heads=2,
                                           context_in_dim=64, axes_dims_rope=(8, 12, 12)).to(torch.bfloat16)
    else:
        tr = QwenImage21Transformer2DModel.from_pretrained(args.model, subfolder="transformer", torch_dtype=torch.bfloat16)
    tr.requires_grad_(False)
    if args.grad_ckpt:
        tr.enable_gradient_checkpointing()
    tr.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.alpha, init_lora_weights="gaussian", target_modules=LORA_TARGETS))
    for p in tr.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return tr.to("cuda")


def save_checkpoint(tr, opt, sched, step, out_dir, keep=2):
    from diffusers.loaders import QwenImageLoraLoaderMixin
    from peft.utils import get_peft_model_state_dict
    ck = out_dir / f"checkpoint-{step}"
    ck.mkdir(parents=True, exist_ok=True)
    lora_sd = get_peft_model_state_dict(tr)
    QwenImageLoraLoaderMixin.save_lora_weights(ck, transformer_lora_layers={k: v.to(torch.bfloat16) for k, v in lora_sd.items()})
    torch.save({"step": step, "lora": lora_sd, "opt": opt.state_dict(), "sched": sched.state_dict()}, ck / "training_state.pt")
    for old in sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))[:-keep]:
        shutil.rmtree(old)
    return ck


def try_resume(tr, opt, sched, out_dir):
    from peft import set_peft_model_state_dict
    cks = sorted(out_dir.glob("checkpoint-*/training_state.pt"), key=lambda p: int(p.parent.name.split("-")[1]))
    if not cks:
        return 0
    st = torch.load(cks[-1], map_location="cpu", weights_only=False)
    set_peft_model_state_dict(tr, st["lora"])
    opt.load_state_dict(st["opt"])
    sched.load_state_dict(st["sched"])
    print(f"resumed from {cks[-1].parent.name}", flush=True)
    return st["step"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/Qwen-Image-2.1"))
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lambda-mask", type=float, default=0.0)
    ap.add_argument("--sigma-sampling", default="logit_normal", choices=["logit_normal", "uniform"])
    ap.add_argument("--bg-target", default="data", choices=["data", "base"])
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--keep-checkpoints", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    ap.add_argument("--tiny-test", action="store_true")
    args = ap.parse_args()
    t_start = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1), encoding="utf-8")
    torch.manual_seed(args.seed)

    from diffusers.optimization import get_cosine_schedule_with_warmup
    tr = build_transformer(args)
    params = [p for p in tr.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params):,}", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4, eps=1e-8)
    sched = get_cosine_schedule_with_warmup(opt, args.warmup, args.steps)
    step = try_resume(tr, opt, sched, out)
    sched_cfg = {} if args.tiny_test else json.loads((Path(args.model) / "scheduler/scheduler_config.json").read_text())

    ds = CachedLatentDataset(args.cache)
    batches = infinite_loader(ds, args.batch_size, args.seed + step, args.workers)
    gen = torch.Generator("cuda").manual_seed(args.seed + step)
    log_f = open(out / "train_log.jsonl", "a", encoding="utf-8")
    print(f"{len(ds)} samples; start step {step}/{args.steps}; λ={args.lambda_mask}; bg_target={args.bg_target}", flush=True)
    tr.train()
    torch.cuda.reset_peak_memory_stats()
    t_win, acc = time.time(), []
    while step < args.steps:
        for _ in range(args.grad_accum):
            b = next(batches)
            x0 = b["latents"].to("cuda", torch.bfloat16, non_blocking=True)
            emb = b["prompt_embeds"].to("cuda", torch.bfloat16, non_blocking=True)
            emb_mask = None if b["prompt_mask"] is None else b["prompt_mask"].to("cuda")
            bsz, n_tok, _ = x0.shape
            hw = int(math.isqrt(n_tok))
            mu = mu_from_scheduler_config(sched_cfg, n_tok)
            sigma = sample_sigmas(bsz, mu, gen, "cuda", args.sigma_sampling)
            noise = torch.randn(x0.shape, generator=gen, device="cuda", dtype=x0.dtype)
            xt, target = noisy_input_and_target(x0, noise, sigma)
            img_mask = torch.cat([torch.zeros(bsz, emb.shape[1], dtype=torch.bool, device="cuda"),
                                  torch.ones(bsz, n_tok // 4, dtype=torch.bool, device="cuda")], dim=1)
            fwd = dict(hidden_states=xt, encoder_hidden_states=emb, timestep=sigma.to(torch.bfloat16),
                       img_shapes=[[(1, hw, hw)]] * bsz, img_mask=img_mask, encoder_hidden_states_mask=emb_mask,
                       return_dict=False)
            tok_mask = b["token_mask"].to("cuda")
            extra = {}
            if args.bg_target == "base":  # đích ngoài vùng chữ = dự đoán của model gốc
                with torch.no_grad():
                    tr.disable_adapters()
                    v_base = tr(**fwd)[0][:, -n_tok:]
                    tr.enable_adapters()
                m3 = tok_mask.to(target.dtype)[..., None]
                with torch.no_grad():
                    extra["mse_base_vs_data_bg"] = (((v_base.float() - target.float()) ** 2).mean(-1) * (1 - m3[..., 0].float())).sum().item() / max(1.0, (1 - m3).sum().item())
                target = m3 * target + (1 - m3) * v_base
            pred = tr(**fwd)[0][:, -n_tok:]
            loss, st = weighted_flow_matching_loss(pred, target, tok_mask, args.lambda_mask)
            st.update(extra)
            (loss / args.grad_accum).backward()
            acc.append({**st, "loss": loss.item(), "sigma": sigma.mean().item()})
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % args.log_every == 0 or step == 1:
            dt = (time.time() - t_win) / (args.log_every if step > 1 else 1)
            rec = {"step": step, "lr": sched.get_last_lr()[0], "grad_norm": round(grad_norm, 4),
                   "sec_per_step": round(dt, 3), "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                   **{k: round(sum(a[k] for a in acc) / len(acc), 5) for k in acc[0]}}
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            print(json.dumps(rec), flush=True)
            t_win, acc = time.time(), []
        out_of_time = args.max_minutes and (time.time() - t_start) / 60 > args.max_minutes
        if step % args.save_every == 0 or step == args.steps or out_of_time:
            ck = save_checkpoint(tr, opt, sched, step, out, args.keep_checkpoints)
            print(f"saved {ck}", flush=True)
            if out_of_time:
                print("TIME_STOP", flush=True)
                return
    final = out / "final"
    if final.exists():
        shutil.rmtree(final)
    shutil.copytree(out / f"checkpoint-{step}", final, ignore=shutil.ignore_patterns("training_state.pt"))
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
