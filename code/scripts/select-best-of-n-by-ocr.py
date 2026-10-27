"""Best-of-N lúc sinh ảnh: mỗi prompt có N ảnh (N seed), chọn ảnh có reward cao nhất theo OCR chọn (SELECTOR),
rồi chấm ảnh được chọn bằng từng OCR. Báo:
- pass@1: trung bình trên mọi ảnh (chất lượng một lần sinh);
- best-of-N (chọn bằng SELECTOR): chấm bằng OCR khác SELECTOR mới là con số không thiên vị;
- oracle: chọn và chấm bằng cùng một OCR (trần trên).

Usage: python scripts/select-best-of-n-by-ocr.py --prompts bench/viglyph/prompts_draft_v0.jsonl --run outputs/bestofn_base
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vitext.metrics.diacritic_aware_text_metrics import score_pair  # noqa: E402


def load_ocr(run: Path, engine: str) -> dict:
    return {json.loads(ln)["image"]: json.loads(ln)["text"] for ln in open(run / f"ocr_{engine}.jsonl", encoding="utf-8")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--selector", default="qwen3vl")
    ap.add_argument("--engines", nargs="+", default=["qwen3vl", "vintern"])
    ap.add_argument("--n", type=int, default=None, help="chỉ dùng N ảnh đầu (theo seed) mỗi prompt")
    args = ap.parse_args()
    run = Path(args.run)
    prompts = {json.loads(ln)["id"]: json.loads(ln) for ln in open(args.prompts, encoding="utf-8")}
    ocr = {e: load_ocr(run, e) for e in args.engines}
    common = set.intersection(*[set(o) for o in ocr.values()])
    cands = defaultdict(list)
    for img in sorted(common, key=lambda s: int(s.rsplit("_s", 1)[1].split(".")[0])):
        cands[img.rsplit("_s", 1)[0]].append(img)
    if args.n:
        cands = {k: v[: args.n] for k, v in cands.items()}
    score = {e: {img: score_pair(prompts[img.rsplit("_s", 1)[0]]["text"], ocr[e][img]) for img in common} for e in args.engines}
    report = {}
    for group in ("all", "short", "medium", "long"):
        ids = [k for k in cands if group == "all" or prompts[k]["bucket"] == group]
        if not ids:
            continue
        row = {"n_prompts": len(ids), "n_per_prompt": round(sum(len(cands[k]) for k in ids) / len(ids), 2)}
        for e in args.engines:
            allimgs = [img for k in ids for img in cands[k]]
            row[f"pass1_{e}"] = round(sum(score[e][i]["exact_span"] for i in allimgs) / len(allimgs), 3)
            pick = [max(cands[k], key=lambda i: (1 - score[args.selector][i]["cer_nfd"], -cands[k].index(i))) for k in ids]
            row[f"bestof_sel-{args.selector}_{e}"] = round(sum(score[e][i]["exact_span"] for i in pick) / len(pick), 3)
            oracle = [max(cands[k], key=lambda i: (1 - score[e][i]["cer_nfd"], -cands[k].index(i))) for k in ids]
            row[f"oracle_{e}"] = round(sum(score[e][i]["exact_span"] for i in oracle) / len(oracle), 3)
        report[group] = row
    (run / f"bestofn_sel-{args.selector}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    keys = [k for k in report["all"] if k not in ("n_prompts",)]
    print("| nhóm | " + " | ".join(keys) + " |")
    print("|---|" + "---|" * len(keys))
    for g, row in report.items():
        print(f"| {g} | " + " | ".join(str(row[k]) for k in keys) + " |")


if __name__ == "__main__":
    main()
