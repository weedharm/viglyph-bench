"""So sánh theo cặp hai run (cùng prompt, cùng seed) dưới nhiều OCR: tỉ lệ đúng, McNemar, sign test CER_NFD
(theo ảnh và theo prompt), phân rã lỗi (chữ cơ sở / dấu thanh / dấu phụ) và tỉ lệ khôi phục từng loại dấu.

Ảnh khớp theo tên file ({id}_s{seed}.png) nên dùng được cho run 1 seed lẫn run nhiều seed (best-of-N).
Chữ đích lấy từ --prompts theo id. Chấm lại trực tiếp từ ocr_<engine>.jsonl (không cần scores_*.jsonl).

Usage: python scripts/compare-two-runs-paired-across-ocrs.py --prompts bench/viglyph/prompts_draft_v0.jsonl \
           --a outputs/bestofn_base --b outputs/bestofn_rl_v3_ck80 --ocrs qwen3vl vintern easyocr \
           --out outputs/paired_base_vs_rl80_4seed.json
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vitext.metrics.diacritic_aware_text_metrics import score_pair  # noqa: E402


def sign_test_p(k: int, n: int) -> float:
    """Kiểm định dấu hai phía (nhị thức, p = 0,5); dùng cho McNemar chính xác trên cặp bất đồng."""
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n) if n else 1.0


def load_scores(run: Path, engine: str, targets: dict) -> dict:
    out = {}
    for ln in open(run / f"ocr_{engine}.jsonl", encoding="utf-8"):
        r = json.loads(ln)
        pid = r["image"].rsplit("_s", 1)[0]
        if pid in targets:
            out[r["image"]] = (pid, score_pair(targets[pid], r["text"]))
    return out


def recovery(scores, keys, total_key, ok_key) -> dict:
    tot, ok = Counter(), Counter()
    for k in keys:
        tot.update(scores[k][1][total_key])
        ok.update(scores[k][1][ok_key])
    return {m: round(ok[m] / tot[m], 4) for m in sorted(tot) if tot[m]}


def compare(a, b, keys) -> dict:
    n = len(keys)
    res = {"n_images": n, "n_prompts": len({a[k][0] for k in keys})}
    for key in ("exact_span", "exact_span_ci"):
        fix = sum(1 for k in keys if not a[k][1][key] and b[k][1][key])
        brk = sum(1 for k in keys if a[k][1][key] and not b[k][1][key])
        res[key] = {"a": round(sum(a[k][1][key] for k in keys) / n, 4), "b": round(sum(b[k][1][key] for k in keys) / n, 4),
                    "fix": fix, "break": brk, "p_mcnemar": sign_test_p(min(fix, brk), fix + brk)}
    better = sum(1 for k in keys if b[k][1]["cer_nfd"] < a[k][1]["cer_nfd"])
    worse = sum(1 for k in keys if b[k][1]["cer_nfd"] > a[k][1]["cer_nfd"])
    pa, pb = defaultdict(float), defaultdict(float)
    for k in keys:
        pa[a[k][0]] += a[k][1]["cer_nfd"]
        pb[b[k][0]] += b[k][1]["cer_nfd"]
    p_up = sum(1 for p in pa if pb[p] < pa[p])
    p_dn = sum(1 for p in pa if pb[p] > pa[p])
    res["cer_nfd"] = {"a": round(sum(a[k][1]["cer_nfd"] for k in keys) / n, 4),
                      "b": round(sum(b[k][1]["cer_nfd"] for k in keys) / n, 4),
                      "better": better, "worse": worse, "p_sign": sign_test_p(min(better, worse), better + worse),
                      "prompt_better": p_up, "prompt_worse": p_dn, "p_prompt": sign_test_p(min(p_up, p_dn), p_up + p_dn)}
    res["errors"] = {part: {"a": sum(a[k][1]["errors"][part] for k in keys), "b": sum(b[k][1]["errors"][part] for k in keys)}
                     for part in ("base", "tone", "diacritic")}
    res["tone_recovery"] = {"a": recovery(a, keys, "tone_total", "tone_ok"), "b": recovery(b, keys, "tone_total", "tone_ok")}
    res["diac_recovery"] = {"a": recovery(a, keys, "diac_total", "diac_ok"), "b": recovery(b, keys, "diac_total", "diac_ok")}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--ocrs", nargs="+", default=["qwen3vl", "vintern", "easyocr"])
    ap.add_argument("--only-ids", default=None, help="file chứa danh sách id (mỗi dòng một id) để chấm tập con")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    targets = {json.loads(ln)["id"]: json.loads(ln)["text"] for ln in open(args.prompts, encoding="utf-8")}
    if args.only_ids:
        keep = {ln.strip() for ln in open(args.only_ids, encoding="utf-8") if ln.strip()}
        targets = {k: v for k, v in targets.items() if k in keep}
    report = {"a": args.a, "b": args.b, "prompts": args.prompts}
    for eng in args.ocrs:
        a, b = load_scores(Path(args.a), eng, targets), load_scores(Path(args.b), eng, targets)
        keys = sorted(set(a) & set(b))
        if not keys:
            print(f"[{eng}] không có ảnh chung")
            continue
        r = report[eng] = compare(a, b, keys)
        e, c = r["exact_span"], r["cer_nfd"]
        err = "  ".join(f"{p} {v['a']}→{v['b']}" for p, v in r["errors"].items())
        print(f"[{eng}] {r['n_images']} ảnh / {r['n_prompts']} prompt | exact {e['a']:.3f}→{e['b']:.3f} "
              f"(sửa {e['fix']}/hỏng {e['break']}, p={e['p_mcnemar']:.2g}) | CER_NFD {c['a']:.4f}→{c['b']:.4f} "
              f"(tốt {c['better']}/kém {c['worse']}, p={c['p_sign']:.2g}; prompt {c['prompt_better']}/{c['prompt_worse']}, "
              f"p={c['p_prompt']:.2g}) | lỗi {err}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print("→", args.out)


if __name__ == "__main__":
    main()
