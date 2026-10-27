"""RL cho chữ có dấu: DiffusionNFT (CFG-free) trên Qwen-Image-2.1, reward OCR nhạy dấu (1 − CER_NFD), 1 GPU.

Mỗi vòng: (1) adapter "old" sinh G ảnh cho mỗi P prompt (768², ít bước, ODE), Qwen3-VL đọc chữ → reward;
(2) chuẩn hoá theo nhóm → xác suất tối ưu p; (3) train adapter "default" bằng loss NFT tại k mức nhiễu mỗi ảnh;
(4) θ_old ← η·θ_old + (1 − η)·θ. Định kỳ chấm tập dev (1 ảnh/prompt, seed cố định) bằng chính OCR reward
để xem đường học; kết quả cuối phải chấm lại bằng OCR khác (Vintern) và nhãn người.
Lưu <out>/ckpt-<it>/ (LoRA định dạng diffusers, dùng được với generate-images --lora) + state để resume.

Usage: python scripts/train-rl-nft-qwen-image-21.py --model models/Qwen-Image-2.1 \
           --prompts bench/selftrain/prompts_rl_v1.jsonl --dev bench/viglyph/prompts_dev_v1.jsonl \
           --embeds data/cache/rl_prompt_embeds_v1.pt --out outputs/rl_nft_v1 --iters 120
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vitext.rl.diffusion_nft_objective import (  # noqa: E402
    diacritic_ocr_reward, nft_loss, nft_loss_text_region, optimality_probability, soft_update_old_adapter)
from vitext.rl.qwen_image_rollout import build_rollout_pipeline, rollout_group  # noqa: E402
from vitext.train.flow_matching_text_mask_loss import (  # noqa: E402
    mu_from_scheduler_config, noisy_input_and_target, sample_sigmas)

LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "img_mlp.gate_layer", "img_mlp.proj", "img_mlp.out"]


def build_policy(args):
    from diffusers import QwenImage21Transformer2DModel
    from peft import LoraConfig, set_peft_model_state_dict
    if args.tiny_test:  # DiT tí hon ngẫu nhiên để thử vòng lặp trên GPU nhỏ
        torch.manual_seed(0)
        tr = QwenImage21Transformer2DModel(num_layers=2, attention_head_dim=32, num_attention_heads=2,
                                           context_in_dim=64, axes_dims_rope=(8, 12, 12)).to(torch.bfloat16)
    else:
        tr = QwenImage21Transformer2DModel.from_pretrained(args.model, subfolder="transformer", torch_dtype=torch.bfloat16)
    tr.requires_grad_(False)
    tr.enable_gradient_checkpointing()
    cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, init_lora_weights="gaussian", target_modules=LORA_TARGETS)
    tr.add_adapter(cfg, adapter_name="default")
    tr.add_adapter(cfg, adapter_name="old")
    for n, p in tr.named_parameters():
        if "lora_" in n:
            p.data = p.data.float()
    if args.init_lora:  # khởi tạo từ LoRA có sẵn (ví dụ RFT), nạp vào cả hai adapter
        from diffusers.loaders import QwenImageLoraLoaderMixin
        sd = QwenImageLoraLoaderMixin.lora_state_dict(args.init_lora)
        sd = sd[0] if isinstance(sd, tuple) else sd
        sd = {k.removeprefix("transformer."): v for k, v in sd.items()}
        for name in ("default", "old"):
            set_peft_model_state_dict(tr, sd, adapter_name=name)
    copy_adapter(tr, "default", "old")
    tr.set_adapter("default")
    return tr.to("cuda")


@torch.no_grad()
def copy_adapter(tr, src, dst):
    params = dict(tr.named_parameters())
    for n, p in params.items():
        if f".{src}." in n and "lora_" in n:
            params[n.replace(f".{src}.", f".{dst}.")].data.copy_(p.data)


def predict(tr, xt, emb, sigma, hw):
    bsz, n_tok, _ = xt.shape
    img_mask = torch.cat([torch.zeros(bsz, emb.shape[1], dtype=torch.bool, device="cuda"),
                          torch.ones(bsz, n_tok // 4, dtype=torch.bool, device="cuda")], dim=1)
    return tr(hidden_states=xt, encoder_hidden_states=emb.expand(bsz, -1, -1), timestep=sigma.to(torch.bfloat16),
              img_shapes=[[(1, hw, hw)]] * bsz, img_mask=img_mask, return_dict=False)[0][:, -n_tok:]


class FakeOCR:
    """Chỉ cho --tiny-test: trả về chữ đích bị xoá ngẫu nhiên vài ký tự, để reward có phương sai."""

    def __init__(self, rows):
        self.texts, self.rng = [r["text"] for r in rows], random.Random(0)

    def read_batch(self, images):
        out = []
        for _ in images:
            t = self.rng.choice(self.texts)
            out.append("".join(ch for ch in t if self.rng.random() > 0.15))
        return out


def save_ckpt(tr, opt, it, out, rng_state):
    from diffusers.loaders import QwenImageLoraLoaderMixin
    from peft.utils import get_peft_model_state_dict
    ck = out / f"ckpt-{it}"
    ck.mkdir(parents=True, exist_ok=True)
    sd = get_peft_model_state_dict(tr, adapter_name="default")
    QwenImageLoraLoaderMixin.save_lora_weights(ck, transformer_lora_layers={k: v.to(torch.bfloat16) for k, v in sd.items()})
    torch.save({"it": it, "default": sd, "old": get_peft_model_state_dict(tr, adapter_name="old"),
                "opt": opt.state_dict(), "rng": rng_state}, ck / "rl_state.pt")
    return ck


def try_resume(tr, opt, out):
    from peft import set_peft_model_state_dict
    cks = sorted(out.glob("ckpt-*/rl_state.pt"), key=lambda p: int(p.parent.name.split("-")[1]))
    if not cks:
        return 0, None
    st = torch.load(cks[-1], map_location="cpu", weights_only=False)
    set_peft_model_state_dict(tr, st["default"], adapter_name="default")
    set_peft_model_state_dict(tr, st["old"], adapter_name="old")
    tr.set_adapter("default")
    opt.load_state_dict(st["opt"])
    print(f"resumed from {cks[-1].parent.name}", flush=True)
    return st["it"], st["rng"]


@torch.no_grad()
def eval_dev(tr, pipe, ocr, dev_rows, embeds, args, ocr2=None):
    tr.set_adapter("default")
    recs, recs2 = [], []
    for i in range(0, len(dev_rows), args.group):
        chunk = dev_rows[i:i + args.group]
        imgs = []
        for r in chunk:
            _, ims = rollout_group(pipe, embeds[r["id"]], 1, args.res, args.rollout_steps, [r["seed"]])
            imgs += ims
        for r, t in zip(chunk, ocr.read_batch(imgs)):
            recs.append(diacritic_ocr_reward(r["text"], t))
        if ocr2 is not None:
            recs2 += [diacritic_ocr_reward(r["text"], t) for r, t in zip(chunk, ocr2.read_batch(imgs))]
    res = {"dev_reward": round(sum(x["reward"] for x in recs) / len(recs), 4),
           "dev_exact": round(sum(x["exact_span"] for x in recs) / len(recs), 4)}
    if recs2:
        res.update({"dev_reward2": round(sum(x["reward"] for x in recs2) / len(recs2), 4),
                    "dev_exact2": round(sum(x["exact_span"] for x in recs2) / len(recs2), 4)})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--embeds", required=True)
    ap.add_argument("--ocr-model", default=str(ROOT / "models/Qwen3-VL-8B-Instruct"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--prompts-per-iter", type=int, default=6)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--res", type=int, default=768)
    ap.add_argument("--rollout-steps", type=int, default=16)
    ap.add_argument("--timesteps-per-sample", type=int, default=2)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--opt-steps-per-iter", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--eta-rate", type=float, default=0.001)
    ap.add_argument("--eta-max", type=float, default=0.5)
    ap.add_argument("--init-lora", default=None)
    ap.add_argument("--text-region-loss", action="store_true",
                    help="Qwen3-VL định vị khung chữ; NFT chỉ trên token vùng chữ, ngoài vùng chữ giữ θ sát θ_old")
    ap.add_argument("--w-keep", type=float, default=1.0)
    ap.add_argument("--w-base", type=float, default=0.0, help="phạt ‖v_θ − v_gốc‖² trên mọi token (LoRA tắt)")
    ap.add_argument("--symmetric-p", action="store_true", help="căn trung bình p mỗi nhóm về đúng 0.5")
    ap.add_argument("--no-negative", action="store_true",
                    help="p = max(p, 0.5): chỉ học từ ảnh tốt hơn trung bình nhóm; ảnh kém chỉ giữ θ gần θ_old")
    ap.add_argument("--reward-ocr2", choices=["none", "easyocr"], default="none",
                    help="OCR thứ hai không tự sửa dấu; reward = (1 − mix)·r_Qwen3VL + mix·r_OCR2")
    ap.add_argument("--reward-mix", type=float, default=0.5)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--seed", type=int, default=20261401)
    ap.add_argument("--tiny-test", action="store_true", help="DiT tí hon + OCR giả + embedding giả (thử code)")
    args = ap.parse_args()
    t_start, out = time.time(), Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1), encoding="utf-8")

    from vitext.ocr.qwen3vl_ocr_wrapper import Qwen3VLOCR, boxes_to_token_mask, locate_text_boxes_batch
    from PIL import ImageDraw
    rows = [json.loads(ln) for ln in open(args.prompts, encoding="utf-8")]
    dev_rows = [json.loads(ln) for ln in open(args.dev, encoding="utf-8")]
    if args.tiny_test:
        g = torch.Generator().manual_seed(1)
        embeds = {r["id"]: torch.randn(12, 64, generator=g).bfloat16() for r in rows + dev_rows}
    else:
        embeds = torch.load(args.embeds, map_location="cpu", weights_only=True)
    tr = build_policy(args)
    params = [p for n, p in tr.named_parameters() if "lora_" in n and ".default." in n]
    print(f"trainable (default adapter): {sum(p.numel() for p in params):,}", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    start, rng_state = try_resume(tr, opt, out)
    pipe = build_rollout_pipeline(args.model, tr)
    ocr = FakeOCR(rows + dev_rows) if args.tiny_test else Qwen3VLOCR(args.ocr_model, args.ocr_model, "qwen3vl-8b-instruct")
    ocr2 = None
    if args.reward_ocr2 == "easyocr":
        from vitext.ocr.easyocr_line_reader import EasyOCRReader
        ocr2 = FakeOCR(rows + dev_rows) if args.tiny_test else EasyOCRReader()
    sched_cfg = json.loads((Path(args.model) / "scheduler/scheduler_config.json").read_text())
    hw = args.res // 16
    mu = mu_from_scheduler_config(sched_cfg, hw * hw)
    rng = random.Random(args.seed)
    if rng_state:
        rng.setstate(rng_state)
    gen = torch.Generator("cuda").manual_seed(args.seed + start)
    log_f = open(out / "rl_log.jsonl", "a", encoding="utf-8")
    if start == 0:
        rec = {"it": 0, **eval_dev(tr, pipe, ocr, dev_rows, embeds, args, ocr2)}
        log_f.write(json.dumps(rec) + "\n"), log_f.flush()
        print(json.dumps(rec), flush=True)

    for it in range(start + 1, args.iters + 1):
        t0 = time.time()
        # 1) thu thập bằng policy cũ
        tr.set_adapter("old")
        batch = []
        for g, r in enumerate(rng.sample(rows, args.prompts_per_iter)):
            seeds = [rng.randrange(2**31) for _ in range(args.group)]
            lat, imgs = rollout_group(pipe, embeds[r["id"]], args.group, args.res, args.rollout_steps, seeds)
            texts = ocr.read_batch(imgs)
            texts2 = ocr2.read_batch(imgs) if ocr2 is not None else [None] * len(imgs)
            if not args.text_region_loss:
                boxes = [[] for _ in imgs]
            elif args.tiny_test:  # khung giả: một dải ngang giữa ảnh, thỉnh thoảng rỗng
                boxes = [[(100.0, 400.0, 900.0, 600.0)] if k % 3 else [] for k in range(len(imgs))]
            else:
                boxes = locate_text_boxes_batch(ocr, imgs)
            for j, (im, t, t2) in enumerate(zip(imgs, texts, texts2)):
                rw = diacritic_ocr_reward(r["text"], t)
                if t2 is not None:
                    rw2 = diacritic_ocr_reward(r["text"], t2)
                    rw = {**rw, "reward1": rw["reward"], "reward2": rw2["reward"], "exact2": rw2["exact_span"],
                          "reward": (1 - args.reward_mix) * rw["reward"] + args.reward_mix * rw2["reward"]}
                batch.append({"lat": lat[j].detach(), "emb": embeds[r["id"]], "group": g,
                              "mask": boxes_to_token_mask(boxes[j], hw), "n_boxes": len(boxes[j]), **rw})
            if g == 0:
                prev = imgs[0].resize((384, 384))
                d = ImageDraw.Draw(prev)
                for x1, y1, x2, y2 in boxes[0]:
                    d.rectangle((x1 * 0.384, y1 * 0.384, x2 * 0.384, y2 * 0.384), outline=(255, 0, 0), width=2)
                prev.save(out / "samples" / f"it{it:04d}_{r['id']}.jpg", quality=85)
        t1 = time.time()
        rewards = torch.tensor([b["reward"] for b in batch])
        p = optimality_probability(rewards, torch.tensor([b["group"] for b in batch]), symmetric=args.symmetric_p)
        if args.no_negative:
            p = p.clamp(min=0.5)
        # 2) train adapter default
        tr.set_adapter("default")
        tr.train()
        # micro-batch trong cùng một prompt (chung embedding, không cần mask đệm); xáo thứ tự các micro-batch
        chunks = []
        for g in range(args.prompts_per_iter):
            items = [(i, k) for i, b in enumerate(batch) if b["group"] == g for k in range(args.timesteps_per_sample)]
            chunks += [items[c:c + args.micro_batch] for c in range(0, len(items), args.micro_batch)]
        rng.shuffle(chunks)
        per_step = math.ceil(len(chunks) / args.opt_steps_per_iter)
        stats = []
        for c, sub in enumerate(chunks):
            x0s = torch.stack([batch[i]["lat"] for i, _ in sub]).to("cuda", torch.bfloat16)
            embs = batch[sub[0][0]]["emb"][None].to("cuda", torch.bfloat16)
            sigma = sample_sigmas(len(sub), mu, gen, "cuda")
            noise = torch.randn(x0s.shape, generator=gen, device="cuda", dtype=x0s.dtype)
            xt, target = noisy_input_and_target(x0s, noise, sigma)
            with torch.no_grad():
                tr.set_adapter("old")
                v_old = predict(tr, xt, embs, sigma, hw)
                tr.set_adapter("default")  # phải bật lại trước forward có grad (checkpointing tính lại lúc backward)
            v_base = None
            if args.w_base > 0:
                with torch.no_grad():
                    tr.disable_adapters()
                    v_base = predict(tr, xt, embs, sigma, hw)
                    tr.enable_adapters()
                    tr.set_adapter("default")
            v_theta = predict(tr, xt, embs, sigma, hw)
            ps = p[[i for i, _ in sub]].to("cuda")
            if args.text_region_loss:
                masks = torch.stack([batch[i]["mask"] for i, _ in sub]).to("cuda")
                loss, st = nft_loss_text_region(v_theta, v_old, target, ps, masks, args.beta, args.w_keep,
                                                v_base, args.w_base)
            else:
                loss, st = nft_loss(v_theta, v_old, target, ps, args.beta, v_base, args.w_base)
            (loss / per_step).backward()
            stats.append({**st, "loss": loss.item()})
            if (c + 1) % per_step == 0 or c + 1 == len(chunks):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
        eta = soft_update_old_adapter(tr, it, args.eta_rate, args.eta_max)
        grp = [rewards[[j for j, b in enumerate(batch) if b["group"] == g]] for g in range(args.prompts_per_iter)]
        rec = {"it": it, "reward": round(rewards.mean().item(), 4), "exact": round(sum(b["exact_span"] for b in batch) / len(batch), 4),
               "reward_std": round(rewards.std().item(), 4), "zero_var_groups": sum(int(x.std() < 1e-6) for x in grp),
               "eta": round(eta, 4), "sec_rollout": round(t1 - t0, 1), "sec_train": round(time.time() - t1, 1),
               "no_box_frac": round(sum(b["n_boxes"] == 0 for b in batch) / len(batch), 3),
               "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               **{k: round(sum(s[k] for s in stats) / len(stats), 5) for k in stats[0]}}
        if ocr2 is not None:
            rec.update({k: round(sum(b[k] for b in batch) / len(batch), 4) for k in ("reward1", "reward2", "exact2")})
        if it % args.eval_every == 0:
            rec.update(eval_dev(tr, pipe, ocr, dev_rows, embeds, args, ocr2))
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        print(json.dumps(rec), flush=True)
        out_of_time = args.max_minutes and (time.time() - t_start) / 60 > args.max_minutes
        if it % args.save_every == 0 or it == args.iters or out_of_time:
            print(f"saved {save_ckpt(tr, opt, it, out, rng.getstate())}", flush=True)
            if out_of_time:
                print("TIME_STOP", flush=True)
                return
    print("RL_DONE", flush=True)


if __name__ == "__main__":
    main()
