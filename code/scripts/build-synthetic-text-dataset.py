"""Dựng tập ảnh chữ tổng hợp loại A từ kho chữ: PNG 1024², mask vùng chữ, metadata.jsonl (mục 6.4 plan.md).

- Chữ chính theo tỉ lệ 40/40/20 (short/medium/long); chữ dài chỉ vào poster và bảng phấn.
- Chữ phụ lấy từ kho short (70%) và medium (30%). Mỗi chuỗi kho chữ dùng đúng một lần.
- Caption đặt từng chữ đã vẽ trong ngoặc kép; kiểm tra khớp NFC với "texts".

Usage: python scripts/build-synthetic-text-dataset.py --n 10000 --workers 20 --out data/synth
       python scripts/build-synthetic-text-dataset.py --n 20 --out outputs/synth_preview --sheet
"""
import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vitext.data.synthetic_layout_templates import LAYOUT_MAX_TEXTS, LAYOUTS, LONG_TEXT_LAYOUTS  # noqa: E402
from vitext.data.synthetic_render_primitives import FontCatalog, wrap_words  # noqa: E402
from vitext.data.vietnamese_text_corpus_filters import VN_DIACRITIC_CHARS, diacritic_char_counts  # noqa: E402

MAIN_SHARE = {"short": 0.4, "medium": 0.4, "long": 0.2}
N_TEXTS = {1: 0.45, 2: 0.35, 3: 0.20}
_CTX = {}


def plan_jobs(corpus_rows, n, seed):
    rng = random.Random(seed)
    queues = {b: [r for r in corpus_rows if r["bucket"] == b] for b in MAIN_SHARE}
    jobs = []
    for i in range(n):
        b = rng.choices(list(MAIN_SHARE), weights=list(MAIN_SHARE.values()))[0]
        if not queues[b]:
            b = max(queues, key=lambda k: len(queues[k]))
        if not queues[b]:
            break
        main = queues[b].pop()
        layout = rng.choice(LONG_TEXT_LAYOUTS if b == "long" else list(LAYOUTS))
        k = min(rng.choices(list(N_TEXTS), weights=list(N_TEXTS.values()))[0], LAYOUT_MAX_TEXTS[layout])
        k = min(k, 2) if b == "long" else k  # đoạn văn dài: tối đa một dòng phụ
        rows = [main]
        for _ in range(k - 1):
            q = queues["short"] if rng.random() < 0.7 and queues["short"] else queues["medium"]
            if q:
                rows.append(q.pop())
        jobs.append({"idx": i, "layout": layout, "main_bucket": b, "seed": seed * 1000 + i,
                     "texts": [r["text"] for r in rows], "text_ids": [r["id"] for r in rows]})
    left = {b: len(q) for b, q in queues.items()}
    return jobs, left


def _init(fonts_json, out_dir):
    _CTX["catalog"] = FontCatalog(Path(fonts_json))
    _CTX["out"] = Path(out_dir)


def render_job(job):
    rng, nrng = random.Random(job["seed"]), np.random.default_rng(job["seed"])
    long_main = job["main_bucket"] == "long"
    layouts = [job["layout"]] + [x for x in (LONG_TEXT_LAYOUTS if long_main else ("poster", "bien_hieu")) if x != job["layout"]]
    res = None
    for layout in layouts:
        res = LAYOUTS[layout](rng, nrng, _CTX["catalog"], job["texts"], long_main=long_main)
        if res is not None:
            break
    if res is None:
        return {**job, "failed": True}
    quoted = re.findall(r'"([^"]*)"', res["caption"])
    assert quoted == res["texts"] and all(unicodedata.is_normalized("NFC", t) for t in quoted), res["caption"]
    out, stem = _CTX["out"], f"{job['idx']:06d}"
    (out / "A").mkdir(parents=True, exist_ok=True)
    res["image"].save(out / "A" / f"{stem}.png", optimize=False, compress_level=6)
    res["mask"].point(lambda v: 255 if v >= 128 else 0).convert("1").save(out / "A" / f"{stem}_mask.png")
    ids = [job["text_ids"][job["texts"].index(t)] for t in res["texts"]]
    return {"id": f"synA_{stem}", "type": "A", "layout": layout, "image": f"{out.name}/A/{stem}.png",
            "mask": f"{out.name}/A/{stem}_mask.png", "texts": res["texts"], "text_ids": ids,
            "caption": res["caption"], "font": res["font"], "width": 1024, "height": 1024,
            "boxes": [list(b) if b else None for b in res["boxes"]], "main_bucket": job["main_bucket"],
            "seed": job["seed"], "ocr_check": None}


def contact_sheet(rows, out_dir, path, cols=4, tile=360):
    """Lưới ảnh mẫu kèm caption để người dùng duyệt."""
    from vitext.data.synthetic_render_primitives import load_font
    font = load_font(str(ROOT / "data/fonts/NotoSans%5Bwdth%2Cwght%5D.ttf"), 15, "Regular", True)
    cap_h = 150
    rows_n = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile, rows_n * (tile + cap_h)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for i, r in enumerate(rows):
        x, y = (i % cols) * tile, (i // cols) * (tile + cap_h)
        sheet.paste(Image.open(Path(out_dir).parent / r["image"]).convert("RGB").resize((tile - 8, tile - 8)), (x + 4, y + 4))
        lines = wrap_words(f"[{r['id'][-4:]} {r['layout']}] {r['caption']}", font, tile - 12)[:8]
        for j, ln in enumerate(lines):
            d.text((x + 6, y + tile + 2 + j * 18), ln, font=font, fill=(20, 20, 20))
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "data/corpus/corpus_v1.jsonl"))
    ap.add_argument("--fonts", default=str(ROOT / "data/fonts/fonts_ok.json"))
    ap.add_argument("--out", default=str(ROOT / "data/synth"))
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--sheet", action="store_true", help="ghép ảnh thành lưới sheet_XX.png (20 ảnh/tấm)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    corpus = [json.loads(ln) for ln in open(args.corpus, encoding="utf-8")]
    jobs, left = plan_jobs(corpus, args.n, args.seed)
    rows, failed = [], []
    with Pool(args.workers, initializer=_init, initargs=(args.fonts, out)) as pool:
        for i, r in enumerate(pool.imap(render_job, jobs, chunksize=4)):
            (failed if r.get("failed") else rows).append(r)
            if (i + 1) % 500 == 0:
                print(f"{i + 1}/{len(jobs)} done, failed {len(failed)}", flush=True)
    with open(out / "metadata.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    counts = diacritic_char_counts(t for r in rows for t in r["texts"])
    stats = {"n": len(rows), "failed": len(failed), "corpus_left": left,
             "layout": dict(Counter(r["layout"] for r in rows)), "main_bucket": dict(Counter(r["main_bucket"] for r in rows)),
             "n_texts": dict(Counter(len(r["texts"]) for r in rows)), "font": dict(Counter(r["font"].split("-")[0] for r in rows)),
             "min_char_count": min(counts[c] for c in VN_DIACRITIC_CHARS),
             "under_200": {c: counts[c] for c in VN_DIACRITIC_CHARS if counts[c] < 200}}
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    if failed:
        (out / "failed.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in failed), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "font"}, ensure_ascii=False))
    if args.sheet:
        for s in range(0, len(rows), 20):
            contact_sheet(rows[s:s + 20], out, out / f"sheet_{s // 20:02d}.png")


if __name__ == "__main__":
    main()
