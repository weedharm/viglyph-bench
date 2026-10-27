"""Trích ứng viên chuỗi chữ từ Wikipedia tiếng Việt (parquet của wikimedia/wikipedia 20231101.vi).

Hai lượt, chạy song song theo lô bài viết:
1. count_templates: đếm khoá mẫu câu (template_key) để biết câu nào do bot sinh hàng loạt.
2. extract_candidates: lấy tiêu đề, đề mục, câu, mệnh đề, cụm 1–3 từ; bỏ câu có khuôn lặp quá nhiều.
Bài ngắn (< min_chars) gần như toàn bài sơ khai do bot tạo nên bỏ từ đầu.
"""
import hashlib
import random
import re
import unicodedata
from collections import Counter
from multiprocessing import Pool

import pyarrow.parquet as pq

from vitext.data.vietnamese_text_corpus_filters import (
    has_only_allowed_chars, length_bucket, looks_vietnamese, template_key)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[^\W\d_])")
_CLAUSE_SPLIT = re.compile(r"\s*[,;:]\s+")
_TITLE_PAREN = re.compile(r"\s*\([^)]*\)\s*$")
_EDGE_PUNCT = " .,;:!?-–"
_TEMPLATE_COUNTS: Counter = Counter()  # gán trước khi fork worker lượt 2


def _hash(key: str) -> int:
    return int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "little")


def _clean(s: str) -> str:
    s = re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip()
    return re.sub(r"\(\s+", "(", re.sub(r"\s+([,.;:!?)])", r"\1", s))  # "Dương , Lăng ." -> "Dương, Lăng."


def iter_article_batches(paths, min_chars: int, batch_size: int = 2000):
    """Sinh các lô [(id, title, text)] chỉ gồm bài dài >= min_chars."""
    for path in paths:
        f = pq.ParquetFile(path)
        for b in f.iter_batches(batch_size=batch_size, columns=["id", "title", "text"]):
            ids, titles, texts = (b.column(i).to_pylist() for i in range(3))
            yield [(i, t, x) for i, t, x in zip(ids, titles, texts) if len(x) >= min_chars]


def _lines(text: str):
    """Tách bài thành ("heading", dòng) hoặc ("sentence", câu)."""
    for line in text.split("\n"):
        line = _clean(line)
        if not line:
            continue
        if line[-1] not in ".!?:" and len(line.split()) <= 10:
            yield "heading", line
        else:
            for sent in _SENT_SPLIT.split(line):
                yield "sentence", sent


def _count_batch(batch) -> Counter:
    counts = Counter()
    for _, _, text in batch:
        for kind, s in _lines(text):
            if kind == "sentence" and len(s.split()) >= 4:
                counts[_hash(template_key(s))] += 1
    return counts


def count_templates(paths, min_chars: int, workers: int) -> Counter:
    total = Counter()
    with Pool(workers) as pool:
        for c in pool.imap_unordered(_count_batch, iter_article_batches(paths, min_chars)):
            total.update(c)
    return total


def _ok(s: str) -> bool:
    return bool(s) and s.count("(") == s.count(")") and has_only_allowed_chars(s) and looks_vietnamese(s)


def _extract_batch(args):
    batch, max_template_count, caps, seed = args
    out = []
    for art_id, title, text in batch:
        rng = random.Random(f"{seed}:{art_id}")
        src = {"article_id": art_id, "article_title": title}
        t = _clean(_TITLE_PAREN.sub("", title))
        if _ok(t) and length_bucket(len(t.split())) in ("short", "medium"):
            out.append({"text": t, "kind": "title", **src})
        sents, clauses, spans = [], [], []
        for kind, s in _lines(text):
            if kind == "heading":
                if _ok(s):
                    out.append({"text": s.strip(_EDGE_PUNCT), "kind": "heading", **src})
                continue
            n = len(s.split())
            if n >= 4 and _TEMPLATE_COUNTS[_hash(template_key(s))] > max_template_count:
                continue  # câu khuôn bot, lặp hàng loạt
            if _ok(s) and n >= 4 and length_bucket(n) != "too_long":
                sents.append(s.rstrip(",;: "))  # câu dẫn kết thúc bằng ":" -> bỏ dấu treo
            for cl in _CLAUSE_SPLIT.split(s.rstrip(".!?")):
                cl = _clean(cl).strip(_EDGE_PUNCT)
                words = cl.split()
                if 4 <= len(words) <= 10 and _ok(cl):
                    clauses.append(cl)
                if len(words) >= 2:
                    k = rng.randint(1, min(3, len(words)))
                    i = rng.randint(0, len(words) - k)
                    span = " ".join(words[i:i + k]).strip(_EDGE_PUNCT)
                    if _ok(span):
                        spans.append(span)
        for kind, pool, cap in (("sentence", sents, caps["sentence"]), ("clause", clauses, caps["clause"]),
                                ("span", spans, caps["span"])):
            for s in rng.sample(pool, min(cap, len(pool))):
                out.append({"text": s, "kind": kind, **src})
    return out


def extract_candidates(paths, min_chars: int, template_counts: Counter, max_template_count: int,
                       caps: dict, seed: int, workers: int) -> list:
    """Trả về list ứng viên {text, kind, article_id, article_title}; chưa khử trùng."""
    global _TEMPLATE_COUNTS
    _TEMPLATE_COUNTS = template_counts  # worker fork kế thừa biến toàn cục này
    jobs = ((b, max_template_count, caps, seed) for b in iter_article_batches(paths, min_chars))
    out = []
    with Pool(workers) as pool:
        for part in pool.imap(_extract_batch, jobs):
            out.extend(part)
    return out
