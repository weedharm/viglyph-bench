"""Tổng hợp chấm mù bằng mắt (visual audit) cho cặp ảnh gốc / RL: giải mã ảnh 1/2 bằng key.json, gộp hai lượt
chấm độc lập (A, B) và lượt phân xử (C, chỉ cho ca A/B bất đồng), rồi báo:
- tỉ lệ đúng hoàn toàn / sai dấu / sai chữ của từng model, kiểm định McNemar trên cặp ảnh;
- ưa thích (ảnh nào đúng hơn);
- độ đồng thuận giữa hai lượt (tỉ lệ khớp nhãn, Cohen's kappa);
- mức khớp của từng OCR (exact_span_ci) với nhãn "ok" của người/mô hình chấm: độ chính xác khi OCR báo đúng
  (precision) và tỉ lệ ảnh sai bị OCR báo đúng (lỗi bị che).

Ảnh "1" = cột trái (l) và "2" = cột phải (r) của trang chấm human_eval_v1.
Usage: python scripts/score-blind-visual-audit.py --key outputs/human_eval_v1_key.json --audit-dir <dir>/out \
           --prompts bench/viglyph/prompts_draft_v0.jsonl --out outputs/visual_audit_v1.json
"""
import argparse
import json
import sys
from collections import Counter
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vitext.metrics.diacritic_aware_text_metrics import score_pair  # noqa: E402

LABELS = ("ok", "diac", "char")


def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n) if n else 1.0


def cohen_kappa(pairs) -> float:
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in LABELS) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def load_pass(files):
    out = {}
    for f in files:
        for ln in open(f, encoding="utf-8"):
            if ln.strip():
                r = json.loads(ln)
                out[r["id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--audit-dir", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    key = json.load(open(args.key, encoding="utf-8"))
    d = Path(args.audit_dir)
    pa, pb = load_pass(sorted(d.glob("passA_*.jsonl"))), load_pass(sorted(d.glob("passB_*.jsonl")))
    pc = load_pass(sorted(d.glob("passC_*.jsonl")))  # phân xử: {"id", "img1"/"img2" (chỉ ảnh bất đồng), ...}
    ids = sorted(set(pa) & set(pb) & set(key["items"]))
    # độ đồng thuận A/B theo từng ảnh
    agree_pairs = [(pa[i][s]["label"], pb[i][s]["label"]) for i in ids for s in ("img1", "img2")]
    disagree = [(i, s) for i in ids for s in ("img1", "img2") if pa[i][s]["label"] != pb[i][s]["label"]]
    # nhãn cuối: A khi A = B; ngược lại lấy C nếu có, nếu chưa có thì để None
    final = {}
    for i in ids:
        for s in ("img1", "img2"):
            la, lb = pa[i][s]["label"], pb[i][s]["label"]
            final[(i, s)] = la if la == lb else (pc.get(i, {}).get(s, {}).get("label"))
    missing = [k for k, v in final.items() if v is None]
    # giải mã sang model A (gốc) / B (RL)
    side = {"img1": "l", "img2": "r"}
    lab = {}  # (id, model) -> label
    for i in ids:
        meta = key["items"][i]
        for s in ("img1", "img2"):
            lab[(i, meta[side[s]])] = final[(i, s)]
    full = [i for i in ids if lab[(i, "A")] and lab[(i, "B")]]
    rep = {"n_items": len(ids), "n_final": len(full), "agreement": {
        "raw": round(sum(a == b for a, b in agree_pairs) / len(agree_pairs), 3),
        "kappa": round(cohen_kappa(agree_pairs), 3), "n_disagree": len(disagree), "n_unresolved": len(missing)}}
    for m in ("A", "B"):
        c = Counter(lab[(i, m)] for i in full)
        rep[m] = {k: round(c[k] / len(full), 3) for k in LABELS} | {"counts": dict(c)}
        for b in ("short", "medium", "long"):
            sub = [i for i in full if key["items"][i]["bucket"] == b]
            if sub:
                rep[m][f"ok_{b}"] = round(sum(lab[(i, m)] == "ok" for i in sub) / len(sub), 3)
    only_a = sum(lab[(i, "A")] == "ok" and lab[(i, "B")] != "ok" for i in full)
    only_b = sum(lab[(i, "B")] == "ok" and lab[(i, "A")] != "ok" for i in full)
    rep["mcnemar_ok"] = {"only_base_ok": only_a, "only_rl_ok": only_b, "p": mcnemar_p(only_a, only_b)}
    # ưa thích theo lượt A (giải mã)
    pref = Counter()
    for i in ids:
        p = pa[i].get("pref")
        meta = key["items"][i]
        pref[{"1": meta["l"], "2": meta["r"], "eq": "="}.get(p, "?")] += 1
    rep["preference_passA"] = dict(pref)
    pa_b, pb_b = pref.get("B", 0), pref.get("A", 0)
    rep["preference_sign_p"] = mcnemar_p(pa_b, pb_b)
    # CER_NFD và phân rã lỗi tính trên bản chép tay "drawn" (trung bình hai lượt A, B) → thước đo mịn hơn "ok"
    cer, errs = {}, {m: Counter() for m in ("A", "B")}
    for i in ids:
        meta, tgt = key["items"][i], key["items"][i]["text"]
        for s in ("img1", "img2"):
            m = meta[side[s]]
            scores = [score_pair(tgt, ps[i][s]["drawn"], semi_global=False) for ps in (pa, pb)]
            cer[(i, m)] = sum(x["cer_nfd"] for x in scores) / 2
            for x in scores:
                errs[m].update({k: v / 2 for k, v in x["errors"].items()})
    better = sum(cer[(i, "B")] < cer[(i, "A")] for i in ids)
    worse = sum(cer[(i, "B")] > cer[(i, "A")] for i in ids)
    rep["audit_cer_nfd"] = {"A": round(sum(cer[(i, "A")] for i in ids) / len(ids), 4),
                            "B": round(sum(cer[(i, "B")] for i in ids) / len(ids), 4),
                            "rl_better": better, "rl_worse": worse, "p_sign": mcnemar_p(better, worse),
                            "errors_A": {k: round(v, 1) for k, v in errs["A"].items()},
                            "errors_B": {k: round(v, 1) for k, v in errs["B"].items()}}
    # OCR so với nhãn chấm mắt
    prompts = {json.loads(ln)["id"]: json.loads(ln) for ln in open(args.prompts, encoding="utf-8")}
    runs = {"A": Path(key["A"]), "B": Path(key["B"])}
    ocr_rep = {}
    for eng in ("qwen3vl", "vintern", "easyocr"):
        tp = fp = fn = tn = 0
        for m, run in runs.items():
            f = ROOT / run / f"ocr_{eng}.jsonl"
            if not f.exists():
                continue
            texts = {json.loads(ln)["image"]: json.loads(ln)["text"] for ln in open(f, encoding="utf-8")}
            for i in full:
                p = prompts[key["items"][i]["prompt_id"]]
                t = texts.get(f"{p['id']}_s{p['seed']}.png")
                if t is None:
                    continue
                ocr_ok = score_pair(p["text"], t)["exact_span_ci"]
                true_ok = lab[(i, m)] == "ok"
                tp += ocr_ok and true_ok
                fp += ocr_ok and not true_ok
                fn += (not ocr_ok) and true_ok
                tn += (not ocr_ok) and not true_ok
        if tp + fp + fn + tn:
            ocr_rep[eng] = {"ocr_says_ok": tp + fp, "precision": round(tp / max(1, tp + fp), 3),
                            "masked_error_rate": round(fp / max(1, fp + tn), 3), "recall": round(tp / max(1, tp + fn), 3),
                            "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    rep["ocr_vs_audit"] = ocr_rep
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    if disagree:
        print("bất đồng A/B:", " ".join(f"{i}:{s}" for i, s in disagree))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
