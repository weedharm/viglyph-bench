"""Dựng prompt cho tự huấn luyện bằng lọc OCR (rejection-sampling fine-tuning, RFT).

Model sinh ảnh từ các prompt này; chỉ giữ ảnh mà mọi OCR đều đọc đúng chữ đích, rồi fine-tune trên chính các ảnh
đó → học thêm từ mẫu đúng dấu mà không trôi phong cách (ảnh là của chính model).
- Câu mô tả cảnh dùng bộ template RIÊNG, khác hẳn template của ViGlyph-Bench (không học khuôn câu của bộ test).
- Chữ lấy từ kho chữ train (đã loại chuỗi trùng bench), bỏ chuỗi của tập dev; tỉ lệ 40/40/20; seed riêng.

Usage: python scripts/build-self-training-prompts.py --n 768 --out bench/selftrain/prompts_rft_v1.jsonl
       python scripts/build-self-training-prompts.py --n 2000 --share 0.25,0.5,0.25 --seed 20261301 --prefix rl1 \
           --out bench/selftrain/prompts_rl_v1.jsonl     # bộ prompt cho RL (nghiêng về chữ vừa/dài)
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_SEED = 20261209
SCENE_TEMPLATES = [
    "Ảnh chụp mặt tiền một quán ăn nhỏ ở Sài Gòn, biển hiệu gỗ sơn ghi dòng chữ \"{t}\".",
    "Tấm bảng đèn LED trước cửa hàng tạp hoá hiển thị dòng chữ \"{t}\".",
    "Bìa một cuốn tạp chí đặt trên bàn, tiêu đề lớn in chữ \"{t}\".",
    "Áo thun trắng treo trên móc, mặt trước in dòng chữ \"{t}\".",
    "Bao bì túi giấy kraft của tiệm bánh, in dòng chữ \"{t}\" ở giữa.",
    "Biển chỉ dẫn màu xanh lá bên đường, ghi dòng chữ \"{t}\" bằng chữ trắng.",
    "Bức tường sơn trắng trong quán cà phê, có dòng chữ \"{t}\" vẽ tay màu đen.",
    "Cốc giấy cà phê mang đi, in logo và dòng chữ \"{t}\".",
    "Tờ rơi quảng cáo đặt trên quầy, tiêu đề in đậm \"{t}\".",
    "Màn hình điện thoại hiển thị thông báo với dòng chữ \"{t}\".",
]
LONG_SCENE_TEMPLATES = [
    "Trang báo in đặt trên bàn gỗ, đoạn văn in đậm ở giữa trang ghi \"{t}\".",
    "Tấm bảng tin bằng nhựa trắng trong hành lang trường học, in dòng chữ \"{t}\".",
    "Khung tranh treo tường, bên trong là câu trích dẫn \"{t}\" viết bằng chữ đẹp.",
    "Một trang sổ tay mở trên bàn, viết tay dòng chữ \"{t}\" bằng mực xanh.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=768)
    ap.add_argument("--out", default=str(ROOT / "bench/selftrain/prompts_rft_v1.jsonl"))
    ap.add_argument("--share", default="0.4,0.4,0.2", help="tỉ lệ short,medium,long")
    ap.add_argument("--seed", type=int, default=BASE_SEED)
    ap.add_argument("--prefix", default="rft1")
    args = ap.parse_args()
    corpus = [json.loads(ln) for ln in open(ROOT / "data/corpus/corpus_v1.jsonl", encoding="utf-8")]
    dev_ids = {json.loads(ln)["corpus_id"] for ln in open(ROOT / "bench/viglyph/prompts_dev_v1.jsonl", encoding="utf-8")}
    rng = random.Random(args.seed)
    share = dict(zip(("short", "medium", "long"), map(float, args.share.split(","))))
    rows = []
    for b, s in share.items():
        pool = [r for r in corpus if r["bucket"] == b and r["id"] not in dev_ids]
        for r in rng.sample(pool, round(args.n * s)):
            tpls = LONG_SCENE_TEMPLATES if b == "long" else SCENE_TEMPLATES
            rows.append({"text": r["text"], "bucket": b, "corpus_id": r["id"], "prompt": rng.choice(tpls).format(t=r["text"])})
    rng.shuffle(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"id": f"{args.prefix}_{i:04d}", **r, "n_words": len(r["text"].split()),
                                "is_upper": r["text"].isupper(), "seed": args.seed + i}, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} prompts -> {out}")


if __name__ == "__main__":
    main()
