"""DiffusionNFT (Zheng et al., ICLR 2026, arXiv 2509.16117) cho flow matching của Qwen-Image-2.1.

- Reward nhạy dấu (plan mục 6.2): r = max(0, 1 − CER_NFD(chuỗi đích, đoạn OCR khớp tốt nhất)).
- Chuẩn hoá theo nhóm cùng prompt: r_norm = r − mean_nhóm; xác suất tối ưu p = 0.5 + 0.5·clip(r_norm / Z, −1, 1),
  Z = độ lệch chuẩn reward của cả lô (tối thiểu z_min để tránh chia cho số rất nhỏ).
- Loss: L = p·‖v⁺ − u‖² + (1 − p)·‖v⁻ − u‖², với v⁺ = (1 − β)·v_old + β·v_θ, v⁻ = (1 + β)·v_old − β·v_θ,
  u = ε − x0 (đích vận tốc, cùng quy ước với SFT). Mỗi mẫu chia cho stop-grad độ lớn sai số (adaptive weighting).
- Policy cũ (sinh dữ liệu) là adapter LoRA "old", cập nhật mềm: θ_old ← η·θ_old + (1 − η)·θ, η_i = min(η_rate·i, η_max).
"""
import torch

from vitext.metrics.diacritic_aware_text_metrics import score_pair


def diacritic_ocr_reward(target: str, ocr_text: str) -> dict:
    """Reward ∈ [0, 1] từ CER_NFD trên đoạn OCR khớp tốt nhất (semi-global), kèm cờ đúng hoàn toàn."""
    s = score_pair(target, ocr_text)
    return {"reward": max(0.0, 1.0 - s["cer_nfd"]), "exact_span": bool(s["exact_span"]), "cer_nfd": s["cer_nfd"]}


def optimality_probability(rewards: torch.Tensor, group_ids: torch.Tensor, z_min: float = 0.05,
                           symmetric: bool = False) -> torch.Tensor:
    """rewards (N,), group_ids (N,) → p (N,) ∈ [0, 1]; nhóm mọi ảnh cùng điểm cho p = 0.5 (không học gì).

    symmetric=True: căn lại để trung bình p mỗi nhóm đúng 0.5. Reward lệch (đa số gần 1, vài ảnh rất thấp)
    bị cắt ở −1 làm mean(p) > 0.5 → dư lực "học lại chính ảnh của mình", kéo model đã chưng cất CFG trôi dần.
    """
    centered = rewards.clone()
    for g in group_ids.unique():
        m = group_ids == g
        centered[m] = rewards[m] - rewards[m].mean()
    z = max(float(rewards.std()), z_min)
    adv = (centered / z).clamp(-1.0, 1.0)
    if symmetric:  # trừ trung bình nhóm sau khi cắt, rồi co cả nhóm cùng tỉ lệ để vẫn nằm trong [−1, 1]
        for g in group_ids.unique():
            m = group_ids == g
            a = adv[m] - adv[m].mean()
            adv[m] = a / max(1.0, float(a.abs().max()))
    return 0.5 + 0.5 * adv


def nft_loss(v_theta: torch.Tensor, v_old: torch.Tensor, target: torch.Tensor, p: torch.Tensor, beta: float = 1.0,
             v_base=None, w_base: float = 0.0):
    """v_theta (grad), v_old (không grad), target u: (B, N, C); p: (B,). Trả về (loss, thống kê).

    v_base/w_base: neo vào model gốc trên mọi token (như nft_loss_text_region), dùng cho ablation không có vùng chữ.
    """
    v_pos = (1 - beta) * v_old + beta * v_theta
    v_neg = (1 + beta) * v_old - beta * v_theta
    e_pos = (v_pos.float() - target.float()).pow(2).mean(dim=(1, 2))
    e_neg = (v_neg.float() - target.float()).pow(2).mean(dim=(1, 2))
    with torch.no_grad():  # adaptive weighting: chuẩn hoá theo độ lớn sai số của từng mẫu
        scale = (v_theta.float() - target.float()).abs().mean(dim=(1, 2)).clamp(min=1e-3)
    p = p.to(e_pos.dtype)
    loss = ((p * e_pos + (1 - p) * e_neg) / scale).mean()
    stats = {"e_pos": e_pos.mean().item(), "e_neg": e_neg.mean().item(), "p_mean": p.mean().item()}
    if v_base is not None and w_base > 0:
        base_term = (v_theta.float() - v_base.float()).pow(2).mean(dim=(1, 2))
        loss = loss + w_base * (base_term / scale).mean()
        stats["to_base"] = base_term.mean().item()
    return loss, stats


@torch.no_grad()
def soft_update_old_adapter(model, step: int, eta_rate: float = 0.001, eta_max: float = 0.5,
                            src: str = "default", dst: str = "old") -> float:
    """θ_old ← η·θ_old + (1 − η)·θ trên mọi tham số LoRA; trả về η đã dùng."""
    eta = min(eta_rate * step, eta_max)
    params = dict(model.named_parameters())
    for name, p_src in params.items():
        if f".{src}." in name and "lora_" in name:
            p_dst = params[name.replace(f".{src}.", f".{dst}.")]
            p_dst.data.mul_(eta).add_(p_src.data.to(p_dst.dtype), alpha=1 - eta)
    return eta


def nft_loss_text_region(v_theta, v_old, target, p, text_mask, beta: float = 1.0, w_keep: float = 1.0,
                         v_base=None, w_base: float = 0.0):
    """NFT chỉ trên token vùng chữ; ngoài vùng chữ giữ θ sát θ_old: w_keep·‖v_θ − v_old‖².

    text_mask (B, N) bool. Mẫu không có khung chữ nào → dùng NFT trên toàn ảnh (như nft_loss).
    """
    v_pos = (1 - beta) * v_old + beta * v_theta
    v_neg = (1 + beta) * v_old - beta * v_theta
    e_pos = (v_pos.float() - target.float()).pow(2).mean(-1)
    e_neg = (v_neg.float() - target.float()).pow(2).mean(-1)
    keep = (v_theta.float() - v_old.float()).pow(2).mean(-1)
    m = text_mask.to(e_pos.dtype)
    empty = m.sum(1) == 0
    m = torch.where(empty[:, None], torch.ones_like(m), m)
    with torch.no_grad():
        scale = (((v_theta.float() - target.float()).abs().mean(-1) * m).sum(1) / m.sum(1)).clamp(min=1e-3)
    pc = p.to(e_pos.dtype)[:, None]
    nft = ((pc * e_pos + (1 - pc) * e_neg) * m).sum(1) / m.sum(1)
    bg = 1 - m
    keep_term = (keep * bg).sum(1) / bg.sum(1).clamp(min=1)
    loss = (nft / scale + w_keep * keep_term / scale).mean()
    base_term = torch.zeros(())
    if v_base is not None and w_base > 0:  # neo vào model gốc trên mọi token (giống KL của Flow-GRPO)
        base_term = (v_theta.float() - v_base.float()).pow(2).mean(dim=(1, 2))
        loss = loss + w_base * (base_term / scale).mean()
    return loss, {"e_pos": (e_pos * m).sum().item() / m.sum().item(), "e_neg": (e_neg * m).sum().item() / m.sum().item(),
                  "p_mean": p.float().mean().item(), "keep": keep_term.mean().item(), "text_frac": m.mean().item(),
                  "to_base": base_term.mean().item()}
