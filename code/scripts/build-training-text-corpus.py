"""Dựng kho chữ huấn luyện từ Wikipedia tiếng Việt (mục 6.3 plan.md).

- Bỏ bài sơ khai, câu khuôn bot; chỉ giữ chuỗi ký tự hợp lệ, giống tiếng Việt; NFC.
- Tỉ lệ độ dài 40/40/20 (1–3 / 4–10 / 11–25 từ); lấy dư dấu chồng, hỏi/ngã, họ ă, đ.
- Biến thể hoa/thường (giữ nguyên, viết hoa đầu câu, Title, UPPER) để phủ đủ 67 chữ hoa có dấu.
- Bổ sung đến khi mỗi ký tự có dấu xuất hiện >= --min-char-count lần (nếu kho ứng viên còn).
- Loại chuỗi trùng ViGlyph-Bench: trùng khớp, hoặc chứa/bị chứa chuỗi bench >= 3 từ.

Usage: python scripts/build-training-text-corpus.py --n 16000 --out data/corpus/corpus_v1.jsonl
"""
import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vitext.data.vietnamese_text_corpus_filters import (  # noqa: E402
    VN_DIACRITIC_CHARS, diacritic_char_counts, length_bucket, rarity_weight)
from vitext.data.wikipedia_text_candidate_extraction import count_templates, extract_candidates  # noqa: E402
from vitext.metrics.vietnamese_text_normalization import normalize_text  # noqa: E402

BUCKET_SHARE = {"short": 0.4, "medium": 0.4, "long": 0.2}
KIND_SHARE = {  # nguồn trong từng bucket; thiếu thì lấy bù từ nguồn khác cùng bucket
    "short": {"title": 0.5, "span": 0.35, "heading": 0.15},
    "medium": {"clause": 0.5, "sentence": 0.3, "title": 0.1, "heading": 0.1},
    "long": {"sentence": 1.0},
}
CASE_SHARE = {
    "short": {"orig": 0.15, "sentence": 0.15, "title": 0.4, "upper": 0.3},
    "medium": {"orig": 0.2, "sentence": 0.25, "title": 0.3, "upper": 0.25},
    "long": {"orig": 0.4, "sentence": 0.45, "upper": 0.15},
}


def apply_case(text: str, case: str) -> str:
    if case == "upper":
        return text.upper()
    if case == "title":
        return " ".join(w[:1].upper() + w[1:] for w in text.split())
    if case == "sentence":
        return text[:1].upper() + text[1:]
    return text


class BenchFilter:
    """Loại ứng viên trùng hoặc chồng lấn chuỗi của bộ đánh giá."""

    def __init__(self, paths):
        self.texts = set()
        for p in paths:
            for ln in Path(p).read_text(encoding="utf-8").splitlines():
                if ln.strip() and not ln.startswith("#"):
                    self.texts.add(normalize_text(ln, casefold=True))
        self.long = [b for b in self.texts if len(b.split()) >= 3]

    def leaks(self, text: str) -> bool:
        t = normalize_text(text, casefold=True)
        if t in self.texts or any(b in t for b in self.long):
            return True
        return len(t.split()) >= 3 and any(t in b for b in self.texts)


def weighted_sample(items, k, rng):
    """Efraimidis–Spirakis: lấy k phần tử không lặp, xác suất tỉ lệ rarity_weight."""
    keyed = sorted(items, key=lambda c: rng.random() ** (1.0 / rarity_weight(c["text"])), reverse=True)
    return keyed[:k]


def pick(rng, shares: dict) -> str:
    return rng.choices(list(shares), weights=list(shares.values()))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", default=[str(p) for p in sorted((ROOT / "data/raw").glob("viwiki-*.parquet"))])
    ap.add_argument("--out", default=str(ROOT / "data/corpus/corpus_v1.jsonl"))
    ap.add_argument("--n", type=int, default=16000)
    ap.add_argument("--min-chars", type=int, default=1500)
    ap.add_argument("--max-template-count", type=int, default=20)
    ap.add_argument("--min-char-count", type=int, default=200)
    ap.add_argument("--bench-texts", nargs="*", default=[str(p) for p in (ROOT / "bench/viglyph").glob("*texts*.txt")])
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261008)
    args = ap.parse_args()
    rng, t0 = random.Random(args.seed), time.time()

    tmpl = count_templates(args.shards, args.min_chars, args.workers)
    print(f"[{time.time() - t0:.0f}s] template keys: {len(tmpl)}; > {args.max_template_count}: "
          f"{sum(1 for v in tmpl.values() if v > args.max_template_count)}")
    caps = {"sentence": 6, "clause": 8, "span": 6}
    cands = extract_candidates(args.shards, args.min_chars, tmpl, args.max_template_count, caps, args.seed, args.workers)
    del tmpl
    seen, pool = set(), {"short": [], "medium": [], "long": []}
    for c in cands:
        key = c["text"].casefold()
        b = length_bucket(len(c["text"].split()))
        if key in seen or b not in pool:
            continue
        seen.add(key)
        pool[b].append(c)
    print(f"[{time.time() - t0:.0f}s] candidates {len(cands)} -> unique {len(seen)}:",
          {b: len(v) for b, v in pool.items()}, Counter(c["kind"] for v in pool.values() for c in v))
    del cands, seen

    bench = BenchFilter(args.bench_texts)
    n_leak, chosen = 0, []
    for b, share in BUCKET_SHARE.items():
        want = round(args.n * share)
        by_kind = {k: [c for c in pool[b] if c["kind"] == k] for k in KIND_SHARE[b]}
        got = []
        for k, s in KIND_SHARE[b].items():
            got += weighted_sample(by_kind[k], round(want * s), rng)
        if len(got) < want:  # bù từ phần còn lại của bucket
            ids = {id(c) for c in got}
            got += weighted_sample([c for c in pool[b] if id(c) not in ids], want - len(got), rng)
        for c in got:
            if bench.leaks(c["text"]):
                n_leak += 1
                continue
            case = pick(rng, CASE_SHARE[b])
            chosen.append({**c, "bucket": b, "case": case, "text": apply_case(c["text"], case)})
    used = {(c["article_id"], c["text"].casefold()) for c in chosen}

    # Bổ sung ký tự thiếu: ưu tiên bucket ngắn/trung bình, chữ hoa lấy bằng UPPER
    counts = diacritic_char_counts(c["text"] for c in chosen)
    n_topup = 0
    for ch in sorted(VN_DIACRITIC_CHARS, key=lambda x: counts[x]):
        if counts[ch] >= args.min_char_count:
            continue
        low, case = ch.lower(), "upper" if ch.isupper() else "sentence"
        extra = []  # ưu tiên chuỗi ngắn để giữ tỉ lệ bucket
        for b in ("short", "medium", "long"):
            part = [c for c in pool[b] if low in c["text"]]
            rng.shuffle(part)
            extra += part
        for c in extra:
            if counts[ch] >= args.min_char_count:
                break
            text = apply_case(c["text"], case)
            if (c["article_id"], text.casefold()) in used and case != "upper":
                continue
            if bench.leaks(text):
                continue
            used.add((c["article_id"], text.casefold()))
            chosen.append({**c, "bucket": length_bucket(len(text.split())), "case": case, "text": text, "topup": ch})
            counts.update(diacritic_char_counts([text]))
            n_topup += 1

    rng.shuffle(chosen)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i, c in enumerate(chosen):
            row = {"id": f"c{i:06d}", "text": c["text"], "bucket": c["bucket"], "n_words": len(c["text"].split()),
                   "kind": c["kind"], "case": c["case"], "source": f"viwiki-20231101.vi:{c['article_id']}",
                   "article_title": c["article_title"]}
            if "topup" in c:
                row["topup"] = c["topup"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = diacritic_char_counts(c["text"] for c in chosen)
    under = {ch: counts[ch] for ch in VN_DIACRITIC_CHARS if counts[ch] < args.min_char_count}
    stats = {"n": len(chosen), "bench_leaks_removed": n_leak, "topup_added": n_topup,
             "bucket": dict(Counter(c["bucket"] for c in chosen)), "kind": dict(Counter(c["kind"] for c in chosen)),
             "case": dict(Counter(c["case"] for c in chosen)), "min_char_count": min(counts[ch] for ch in VN_DIACRITIC_CHARS),
             "under_target": under, "char_counts": {ch: counts[ch] for ch in VN_DIACRITIC_CHARS}}
    out.with_suffix(".stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{time.time() - t0:.0f}s] wrote {len(chosen)} -> {out}")
    print({k: v for k, v in stats.items() if k != "char_counts"})


if __name__ == "__main__":
    main()
