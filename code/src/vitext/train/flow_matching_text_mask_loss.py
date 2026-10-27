"""Flow matching cho Qwen-Image-2.1 và loss có trọng số vùng chữ (mục 6.2 plan.md).

Quy ước khớp pipeline diffusers @cff9daf:
- x_σ = (1 − σ)·x0 + σ·ε, đích vận tốc u = ε − x0 (scheduler.step: x_next = x + (σ_next − σ)·v).
- timestep đưa vào transformer chính là σ (pipeline truyền t/1000 với t = 1000σ).
- σ lấy mẫu logit-normal rồi dịch theo mu của scheduler (time shift kiểu mũ, giống lúc suy luận ở 1024²).
- Mask pixel 1024² max-pool 16× -> 64×64 = 4096 token, trải hàng theo thứ tự pack (view(B, C, H·W)).
"""
import math

import torch
import torch.nn.functional as F

VAE_SCALE = 16  # VAE nén 16×, DiT patch_size 1 -> mỗi token phủ 16×16 px


def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15) -> float:
    """Bản sao calculate_shift của pipeline: nội suy tuyến tính mu theo số token ảnh."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    return image_seq_len * m + (base_shift - m * base_seq_len)


def mu_from_scheduler_config(cfg: dict, image_seq_len: int) -> float:
    return calculate_shift(image_seq_len, cfg.get("base_image_seq_len", 256), cfg.get("max_image_seq_len", 4096),
                           cfg.get("base_shift", 0.5), cfg.get("max_shift", 1.15))


def shift_sigma(sigma: torch.Tensor, mu: float) -> torch.Tensor:
    """Time shift kiểu mũ của FlowMatchEulerDiscreteScheduler: σ' = e^mu / (e^mu + (1/σ − 1))."""
    return math.exp(mu) / (math.exp(mu) + (1.0 / sigma - 1.0))


def sample_sigmas(batch_size: int, mu: float, generator=None, device="cpu", mode="logit_normal") -> torch.Tensor:
    if mode == "logit_normal":
        u = torch.sigmoid(torch.randn(batch_size, generator=generator, device=device))
    else:
        u = torch.rand(batch_size, generator=generator, device=device)
    u = u.clamp(1e-4, 1 - 1e-4)
    return shift_sigma(u, mu)


def token_mask_from_pixel_mask(mask: torch.Tensor) -> torch.Tensor:
    """mask (B, 1, H, W) giá trị {0,1} -> (B, H/16·W/16) bằng max-pool: token chạm nét chữ là 1."""
    pooled = F.max_pool2d(mask.float(), kernel_size=VAE_SCALE, stride=VAE_SCALE)
    return pooled.flatten(1)


def noisy_input_and_target(x0: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor):
    """x0, noise: (B, N, C); sigma: (B,). Trả về (x_σ, u = ε − x0)."""
    s = sigma.view(-1, 1, 1).to(x0.dtype)
    return (1 - s) * x0 + s * noise, noise - x0


def weighted_flow_matching_loss(pred: torch.Tensor, target: torch.Tensor, token_mask: torch.Tensor, lam: float):
    """L = (1/N) Σ_i w_i ‖v_i − u_i‖², w_i = 1 + λ·m_i; sai số tính trung bình theo kênh, ở fp32.

    Trả về (loss, thống kê): mse trong vùng chữ và ngoài vùng chữ để theo dõi riêng.
    """
    err = (pred.float() - target.float()).pow(2).mean(dim=-1)  # (B, N)
    m = token_mask.to(err.dtype)
    loss = ((1.0 + lam * m) * err).mean()
    with torch.no_grad():
        n_text = m.sum().clamp(min=1)
        n_bg = (1 - m).sum().clamp(min=1)
        stats = {"mse": err.mean().item(), "mse_text": (err * m).sum().item() / n_text.item(),
                 "mse_bg": (err * (1 - m)).sum().item() / n_bg.item(), "text_frac": m.mean().item()}
    return loss, stats
