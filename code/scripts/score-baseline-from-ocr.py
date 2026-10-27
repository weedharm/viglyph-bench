"""Chấm baseline: ghép prompts + ocr_<engine>.jsonl -> scores + bảng tổng hợp theo bucket.

Ghi <run>/scores_<engine>.jsonl và <run>/summary_<engine>.json; in bảng markdown.
Usage: python score-baseline-from-ocr.py --prompts bench/viglyph/prompts_draft_v0.jsonl --run outputs/baseline_1024
"""
import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vitext.metrics.diacritic_aware_text_metrics import DIACRITIC_NAMES, TONE_NAMES, score_pair  # noqa: E402


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def aggregate(rows: list) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    out = {"n": n}
    for key in ("exact_full", "exact_full_ci", "exact_span", "exact_span_ci"):
        k = sum(r[key] for r in rows)
        out[key] = round(k / n, 3)
        out[key + "_ci95"] = wilson(k, n)
    for key in ("ned", "cer_nfc", "cer_nfd"):
        out[key] = round(sum(r[key] for r in rows) / n, 4)
    errs = Counter()
    tt, to, dt, do = Counter(), Counter(), Counter(), Counter()
    for r in rows:
        errs.update(r["errors"])
        tt.update(r["tone_total"]), to.update(r["tone_ok"]), dt.update(r["diac_total"]), do.update(r["diac_ok"])
    out["errors"] = dict(errs)
    out["tone_recovery"] = {t: round(to[t] / tt[t], 3) if tt[t] else None for t in TONE_NAMES}
    out["diac_recovery"] = {d: round(do[d] / dt[d], 3) if dt[d] else None for d in DIACRITIC_NAMES}
    return out


def combined_strict(run: Path, engines: dict) -> None:
    """Thận trọng: ảnh chỉ tính 'đúng' khi MỌI OCR đều thấy đúng (giảm lỗi bị OCR tự sửa che mất)."""
    common = set.intersection(*[set(e) for e in engines.values()])
    groups = defaultdict(list)
    for im in common:
        ss = [e[im] for e in engines.values()]
        rec = {"exact_span": all(s["exact_span"] for s in ss), "exact_full": all(s["exact_full"] for s in ss),
               "cer_nfd_max": max(s["cer_nfd"] for s in ss), "bucket": ss[0]["bucket"]}
        groups["all"].append(rec)
        groups["bucket:" + rec["bucket"]].append(rec)
    out = {}
    print(f"\n### Kết hợp thận trọng ({' & '.join(engines)}): đúng khi mọi OCR đều đúng\n")
    print("| group | n | exact_span_all [95% CI] | exact_full_all | mean max CER_NFD |")
    print("|---|---|---|---|---|")
    for g in ["all", "bucket:short", "bucket:medium", "bucket:long"]:
        rs = groups.get(g, [])
        if not rs:
            continue
        k = sum(r["exact_span"] for r in rs)
        out[g] = {"n": len(rs), "exact_span_all": round(k / len(rs), 3), "ci95": wilson(k, len(rs)),
                  "exact_full_all": round(sum(r["exact_full"] for r in rs) / len(rs), 3),
                  "cer_nfd_max": round(sum(r["cer_nfd_max"] for r in rs) / len(rs), 4)}
        o = out[g]
        print(f"| {g} | {o['n']} | {o['exact_span_all']:.3f} {o['ci95']} | {o['exact_full_all']:.3f} | {o['cer_nfd_max']:.3f} |")
    (run / "summary_combined_strict.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    prompts = {json.loads(ln)["id"]: json.loads(ln) for ln in open(args.prompts, encoding="utf-8")}
    all_engines = {}
    for ocr_file in sorted(run.glob("ocr_*.jsonl")):
        engine = ocr_file.stem[4:]
        scored = []
        for ln in open(ocr_file, encoding="utf-8"):
            o = json.loads(ln)
            p = prompts[o["image"].rsplit("_s", 1)[0]]
            s = score_pair(p["text"], o["text"])
            s.update({"id": p["id"], "image": o["image"], "bucket": p["bucket"], "is_upper": p["is_upper"],
                      "target": p["text"], "ocr_raw": o["text"]})
            scored.append(s)
        with open(run / f"scores_{engine}.jsonl", "w", encoding="utf-8") as f:
            for s in scored:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        groups = defaultdict(list)
        for s in scored:
            groups["all"].append(s)
            groups["bucket:" + s["bucket"]].append(s)
            groups["upper" if s["is_upper"] else "mixed_case"].append(s)
        summary = {g: aggregate(rs) for g, rs in groups.items()}
        (run / f"summary_{engine}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n### OCR = {engine}\n")
        print("| group | n | exact_full | exact_span [95% CI] | exact_span_ci | NED | CER_NFC | CER_NFD | err base/tone/diac/other |")
        print("|---|---|---|---|---|---|---|---|---|")
        for g in ["all", "bucket:short", "bucket:medium", "bucket:long", "upper", "mixed_case"]:
            a = summary.get(g)
            if not a or not a["n"]:
                continue
            e = a["errors"]
            print(f"| {g} | {a['n']} | {a['exact_full']:.3f} | {a['exact_span']:.3f} {a['exact_span_ci95']} | "
                  f"{a['exact_span_ci']:.3f} | {a['ned']:.3f} | {a['cer_nfc']:.3f} | {a['cer_nfd']:.3f} | "
                  f"{e.get('base', 0)}/{e.get('tone', 0)}/{e.get('diacritic', 0)}/{e.get('other', 0)} |")
        a = summary["all"]
        print("\ntone recovery:", a["tone_recovery"])
        print("diacritic recovery:", a["diac_recovery"])
        all_engines[engine] = {s["image"]: s for s in scored}
    if len(all_engines) >= 2:
        combined_strict(run, all_engines)


if __name__ == "__main__":
    main()
