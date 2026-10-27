"""Dựng tập dev 60 prompt (20 short / 20 medium / 20 long) để chọn checkpoint LoRA mà không chạm vào bộ test.

Chữ lấy từ phần kho chữ KHÔNG dùng trong tập train (không nằm trong text_ids của data/synth/metadata.jsonl);
kho chữ vốn đã loại chuỗi trùng/chồng lấn ViGlyph-Bench. Câu mô tả dùng cùng template với bộ test để chỉ
khác ở chữ đích. Ghi bench/viglyph/prompts_dev_v1.jsonl cùng schema với prompts_draft_v0.jsonl.

Usage: python scripts/build-dev-prompts-from-unused-corpus.py
"""
import importlib.util
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bench_builder", ROOT / "scripts/build-viglyph-draft-prompts.py")
bench_builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench_builder)

DEV_SEED = 20261108
PER_BUCKET = 20


def main():
    corpus = [json.loads(ln) for ln in open(ROOT / "data/corpus/corpus_v1.jsonl", encoding="utf-8")]
    used = {tid for ln in open(ROOT / "data/synth/metadata.jsonl", encoding="utf-8") for tid in json.loads(ln)["text_ids"]}
    unused = [r for r in corpus if r["id"] not in used]
    rng = random.Random(DEV_SEED)
    picked = []
    for b in ("short", "medium", "long"):
        pool = [r for r in unused if r["bucket"] == b]
        picked += rng.sample(pool, min(PER_BUCKET, len(pool)))
    rows = []
    for idx, r in enumerate(picked):
        bucket = bench_builder.bucket_of(len(r["text"].split()))
        templates = bench_builder.LONG_TEMPLATES if bucket == "long" else bench_builder.SIGN_TEMPLATES
        tpl_idx = idx % len(templates)
        rows.append({"id": f"dev1_{idx:04d}", "text": r["text"], "bucket": bucket, "n_words": len(r["text"].split()),
                     "is_upper": r["text"].isupper(), "template": f"{'long' if bucket == 'long' else 'sign'}_{tpl_idx}",
                     "prompt": templates[tpl_idx].format(t=r["text"]), "seed": DEV_SEED + idx,
                     "marks": bench_builder.mark_counts(r["text"]), "corpus_id": r["id"]})
    out = ROOT / "bench/viglyph/prompts_dev_v1.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(unused)} unused corpus strings -> {len(rows)} dev prompts -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
