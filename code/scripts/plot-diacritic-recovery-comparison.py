"""Biểu đồ tỉ lệ khôi phục từng loại dấu (thanh + dấu phụ) cho nhiều model trên cùng bộ test, theo một OCR.

Đọc <run>/summary_<engine>.json (score-baseline-from-ocr.py). Một cột nhóm cho mỗi loại dấu, mỗi model một màu.
Usage: python scripts/plot-diacritic-recovery-comparison.py --engine vintern \
           --runs "Gốc=outputs/baseline_1024" "RL v3 ck80=outputs/eval_rl_nft_v3b_ckpt-80" --out outputs/diacritic_recovery.png
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {"sac": "sắc", "huyen": "huyền", "hoi": "hỏi", "nga": "ngã", "nang": "nặng",
          "mu": "mũ (â ê ô)", "moc": "móc (ơ ư)", "trang": "trăng (ă)", "gach": "gạch (đ)"}
COLORS = ["#9aa5b1", "#2f6db5", "#d0743c", "#3c9d5d"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help='"Tên=đường_dẫn_run"')
    ap.add_argument("--engine", default="vintern")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    runs = [r.split("=", 1) for r in args.runs]
    keys = list(LABELS)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    width = 0.8 / len(runs)
    for k, (name, path) in enumerate(runs):
        s = json.loads((Path(path) / f"summary_{args.engine}.json").read_text())["all"]
        vals = [{**s["tone_recovery"], **s["diac_recovery"]}[key] for key in keys]
        xs = [i + (k - (len(runs) - 1) / 2) * width for i in range(len(keys))]
        ax.bar(xs, vals, width, label=name, color=COLORS[k % len(COLORS)])
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([LABELS[k] for k in keys])
    ax.set_ylim(0.6, 1.02)
    ax.set_ylabel(f"tỉ lệ khôi phục dấu ({args.engine})")
    ax.axvline(4.5, color="#cccccc", lw=1)
    ax.text(2, 0.615, "dấu thanh", ha="center", fontsize=9, color="#555555")
    ax.text(6.5, 0.615, "dấu phụ", ha="center", fontsize=9, color="#555555")
    ax.legend(fontsize=8, ncol=len(runs), loc="upper center", bbox_to_anchor=(0.5, 1.16), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
