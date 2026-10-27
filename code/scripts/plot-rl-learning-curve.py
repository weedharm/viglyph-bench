"""Vẽ đường học RL từ <run>/rl_log.jsonl: reward rollout (trung bình trượt) và điểm dev (OCR reward) theo vòng.

Usage: python scripts/plot-rl-learning-curve.py outputs/rl_nft_v1/rl_log.jsonl [--out outputs/rl_nft_v1/learning_curve.png]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def rolling(xs, k=5):
    return [sum(xs[max(0, i - k + 1):i + 1]) / len(xs[max(0, i - k + 1):i + 1]) for i in range(len(xs))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = [json.loads(ln) for ln in open(args.log, encoding="utf-8")]
    tr = [r for r in rows if "reward" in r]
    dev = [r for r in rows if "dev_reward" in r]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    it = [r["it"] for r in tr]
    axes[0].plot(it, [r["reward"] for r in tr], color="#9db4d6", lw=1, label="reward mỗi vòng")
    axes[0].plot(it, rolling([r["reward"] for r in tr]), color="#1f4e8c", lw=2, label="trung bình trượt 5 vòng")
    axes[0].plot(it, rolling([r["exact"] for r in tr]), color="#c0504d", lw=2, label="tỉ lệ đúng hoàn toàn (trượt 5)")
    axes[0].set_xlabel("vòng RL")
    axes[0].set_title("Rollout (prompt huấn luyện)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    di = [r["it"] for r in dev]
    axes[1].plot(di, [r["dev_reward"] for r in dev], "o-", color="#1f4e8c", label="dev reward (1 − CER_NFD)")
    axes[1].plot(di, [r["dev_exact"] for r in dev], "s-", color="#c0504d", label="dev đúng hoàn toàn")
    axes[1].set_xlabel("vòng RL")
    axes[1].set_title("Tập dev 60 prompt (OCR reward, 768², 16 bước)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out = Path(args.out) if args.out else Path(args.log).with_name("learning_curve.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
