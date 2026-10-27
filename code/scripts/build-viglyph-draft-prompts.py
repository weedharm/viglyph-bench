"""Dựng prompts.jsonl bản nháp ViGlyph-Bench từ danh sách chuỗi chữ + template cảnh.

Caption tiếng Việt, chữ đích đặt trong ngoặc kép (mục 6.4 plan.md). Seed cố định theo id.
Usage: python scripts/build-viglyph-draft-prompts.py [texts.txt] [out.jsonl]
"""
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vitext.metrics.vietnamese_text_normalization import decompose_char  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE_SEED = 20261005

SIGN_TEMPLATES = [
    "Ảnh chụp tấm biển hiệu cửa hàng trên mặt tiền phố cổ, trên biển có dòng chữ \"{t}\" viết bằng chữ lớn, rõ ràng.",
    "Một tấm poster thiết kế phẳng hiện đại, nền màu pastel, ở giữa là dòng chữ lớn \"{t}\".",
    "Bảng phấn đen trong quán cà phê, viết tay bằng phấn trắng dòng chữ \"{t}\".",
    "Biển đèn neon phát sáng trên tường gạch vào ban đêm, hiển thị dòng chữ \"{t}\".",
    "Bìa sách tối giản, tiêu đề in đậm \"{t}\" ở chính giữa.",
    "Một băng rôn vải đỏ căng ngang đường, in chữ vàng \"{t}\".",
    "Tấm thiệp giấy kraft đặt trên bàn gỗ, trên thiệp in dòng chữ \"{t}\" bằng mực đen.",
    "Hộp sản phẩm đặt trên nền trắng, mặt trước in dòng chữ \"{t}\".",
]
LONG_TEMPLATES = [
    "Một tấm áp phích khổ lớn dán trên tường, in đoạn văn \"{t}\" bằng phông chữ rõ ràng, dễ đọc.",
    "Trang giấy trắng đặt trên bàn gỗ, trên đó in đoạn văn \"{t}\" bằng chữ đen rõ nét.",
    "Bảng thông báo màu trắng treo ở sảnh, in dòng chữ \"{t}\".",
    "Slide thuyết trình nền xanh đậm, chữ trắng ghi \"{t}\".",
]


def bucket_of(n_words: int) -> str:
    return "short" if n_words <= 3 else "medium" if n_words <= 10 else "long"


def mark_counts(text: str) -> dict:
    counts = Counter()
    for ch in text:
        _, diac, tone = decompose_char(ch)
        if diac:
            counts["diac_" + diac] += 1
        if tone:
            counts["tone_" + tone] += 1
        if diac and tone:
            counts["stacked"] += 1
    return dict(counts)


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "bench/viglyph/draft-v0-texts.txt"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "bench/viglyph/prompts_draft_v0.jsonl"
    texts = [unicodedata.normalize("NFC", ln.strip()) for ln in src.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.startswith("#")]
    rows, per_bucket = [], Counter()
    for idx, text in enumerate(texts):
        n_words = len(text.split())
        bucket = bucket_of(n_words)
        templates = LONG_TEMPLATES if bucket == "long" else SIGN_TEMPLATES
        tpl_idx = per_bucket[bucket] % len(templates)
        per_bucket[bucket] += 1
        rows.append({
            "id": f"vg0_{idx:04d}", "text": text, "bucket": bucket, "n_words": n_words,
            "is_upper": text.isupper(), "template": f"{'long' if bucket == 'long' else 'sign'}_{tpl_idx}",
            "prompt": templates[tpl_idx].format(t=text), "seed": BASE_SEED + idx, "marks": mark_counts(text),
        })
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    total = Counter()
    for r in rows:
        total.update(r["marks"])
    print(f"wrote {len(rows)} prompts -> {out}; buckets={dict(per_bucket)}")
    print("mark coverage:", dict(sorted(total.items())))


if __name__ == "__main__":
    main()
