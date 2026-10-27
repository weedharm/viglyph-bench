"""Chấm MultiTextEdit theo chữ đích (OCR cả ảnh rồi khớp semi-global), cho ảnh model sửa và ảnh đích do người sửa.

- Thao tác giữ/viết chữ (color, relocation, scaling, change_font, exchange, insert): chữ đích phải có trong ảnh.
  Báo exact_span (phân biệt hoa thường), exact_span_ci, CER_NFD (không phân biệt hoa thường), và "sai dấu": đúng khi bỏ dấu nhưng sai khi giữ dấu.
- text_delete: tỉ lệ xoá thành công (chữ cũ không còn: NED của đoạn khớp tốt nhất < 0,8).
- exchange: thêm tỉ lệ chữ cũ vẫn còn nguyên.
- Ảnh đích của người (refs) chấm cùng cách: cho biết trần của OCR khi ảnh đúng.

Usage: python scripts/score-multitextedit-from-ocr.py --run outputs/mte_base_vi
Đọc <run>/manifest.jsonl, <run>/ocr_<engine>.jsonl (ảnh model), <run>/refs_ocr/ocr_<engine>.jsonl (ảnh đích).
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vitext.data.vietnamese_text_corpus_filters import strip_marks  # noqa: E402
from vitext.metrics.diacritic_aware_text_metrics import score_pair  # noqa: E402
from vitext.metrics.vietnamese_text_normalization import normalize_text  # noqa: E402

KEEP_OPS = ["color_change", "relocation", "scaling", "change_font", "exchange", "insert"]


def contains_ignoring_marks(target: str, pred: str) -> bool:
    t = strip_marks(normalize_text(target)).casefold()
    p = strip_marks(normalize_text(pred)).casefold()
    return re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", p) is not None


def score_row(m: dict, ocr_text: str) -> dict:
    rec = {"id": m["id"], "op": m["editing_method"]}
    if m["expected"]:
        s = score_pair(m["expected"], ocr_text)
        s_ci = score_pair(m["expected"].casefold(), ocr_text.casefold())  # lệnh có thể ghi hoa/thường khác ảnh
        base_ok = contains_ignoring_marks(m["expected"], ocr_text)
        rec.update({"exact_span": s["exact_span"], "exact_span_ci": s["exact_span_ci"], "cer_nfd": s_ci["cer_nfd"],
                    "ned": s_ci["ned"], "base_ok": base_ok, "diacritic_fail": base_ok and not s["exact_span_ci"],
                    "pred_span": s_ci["pred_span"]})
    if m["removed"]:
        r = score_pair(m["removed"], ocr_text)
        rec["removed_still_present"] = r["ned"] >= 0.8
        rec["removed_exact_present"] = r["exact_span_ci"]
    return rec


def summarize(recs: list) -> dict:
    out = {}
    groups = defaultdict(list)
    for r in recs:
        groups[r["op"]].append(r)
        if "exact_span" in r:
            groups["ALL_KEEP_OR_WRITE"].append(r)
    for g, rs in sorted(groups.items()):
        o = {"n": len(rs)}
        ks = [r for r in rs if "exact_span" in r]
        if ks:
            n = len(ks)
            o.update({"exact_span": round(sum(r["exact_span"] for r in ks) / n, 3),
                      "exact_span_ci": round(sum(r["exact_span_ci"] for r in ks) / n, 3),
                      "base_ok": round(sum(r["base_ok"] for r in ks) / n, 3),
                      "diacritic_fail_given_base_ok": round(sum(r["diacritic_fail"] for r in ks) / max(1, sum(r["base_ok"] for r in ks)), 3),
                      "cer_nfd": round(sum(r["cer_nfd"] for r in ks) / n, 4)})
        ds = [r for r in rs if "removed_still_present" in r]
        if ds:
            o["removed_still_present"] = round(sum(r["removed_still_present"] for r in ds) / len(ds), 3)
        out[g] = o
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    manifest = {json.loads(ln)["id"]: json.loads(ln) for ln in open(run / "manifest.jsonl", encoding="utf-8")}
    report = {}
    for label, sub in (("model", run), ("reference", run / "refs_ocr")):
        for ocr_file in sorted(sub.glob("ocr_*.jsonl")):
            engine = ocr_file.stem[4:]
            recs = []
            for ln in open(ocr_file, encoding="utf-8"):
                o = json.loads(ln)
                m = manifest.get(o["image"].removesuffix(".png"))
                if m:
                    recs.append(score_row(m, o["text"]))
            (sub / f"scores_{engine}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs),
                                                       encoding="utf-8")
            report[f"{label}/{engine}"] = summarize(recs)
    (run / "summary_mte.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for key, summ in report.items():
        print(f"\n### {key}\n\n| op | n | exact_span | exact_span_ci | đúng nếu bỏ dấu | sai dấu / đúng-bỏ-dấu | CER_NFD | chữ cũ còn |")
        print("|---|---|---|---|---|---|---|---|")
        for g, o in summ.items():
            print(f"| {g} | {o['n']} | {o.get('exact_span', '')} | {o.get('exact_span_ci', '')} | {o.get('base_ok', '')} | "
                  f"{o.get('diacritic_fail_given_base_ok', '')} | {o.get('cer_nfd', '')} | {o.get('removed_still_present', '')} |")


if __name__ == "__main__":
    main()
