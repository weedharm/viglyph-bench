"""Lọc ảnh tự sinh cho RFT: giữ ảnh mà MỌI OCR đều đọc đúng chữ đích (exact_span, phân biệt hoa thường).

Đọc <run>/scores_<engine>.jsonl (do score-baseline-from-ocr.py ghi) và prompts; ghi <run>/rft_metadata.jsonl
theo định dạng cache-latents-and-prompt-embeds.py dùng (ảnh tương đối với --data-root = <run>, mask = null).

Usage: python scripts/select-correct-self-generated-images.py --prompts bench/selftrain/prompts_rft_v1.jsonl --run outputs/rft_gen_v1
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--engines", nargs="+", default=["qwen3vl", "vintern"])
    args = ap.parse_args()
    run = Path(args.run)
    prompts = {json.loads(ln)["id"]: json.loads(ln) for ln in open(args.prompts, encoding="utf-8")}
    ok = None
    for e in args.engines:
        good = {json.loads(ln)["image"] for ln in open(run / f"scores_{e}.jsonl", encoding="utf-8") if json.loads(ln)["exact_span"]}
        ok = good if ok is None else ok & good
    rows, per_bucket = [], Counter()
    for img in sorted(ok):
        p = prompts[img.rsplit("_s", 1)[0]]
        per_bucket[p["bucket"]] += 1
        rows.append({"id": p["id"], "image": f"images/{img}", "mask": None, "caption": p["prompt"], "texts": [p["text"]],
                     "bucket": p["bucket"]})
    (run / "rft_metadata.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    total = Counter(p["bucket"] for p in prompts.values())
    print(f"kept {len(rows)}/{len(prompts)}:", {b: f"{per_bucket[b]}/{total[b]}" for b in total})


if __name__ == "__main__":
    main()
