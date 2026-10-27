"""Chấm test "lỗi cài sẵn": OCR chép trung thực chuỗi sai dấu, hay tự sửa về chuỗi đúng?

faithful   : OCR == chuỗi hiển thị (đúng như ảnh)
autocorrect: OCR == chuỗi gốc (OCR đã "sửa hộ" lỗi dấu -> che lỗi của mô hình sinh ảnh)
other      : khác cả hai
Usage: python score-ocr-injection-test.py outputs/ocr_injection
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vitext.metrics.vietnamese_text_normalization import normalize_text  # noqa: E402


def main():
    root = Path(sys.argv[1])
    meta = {json.loads(ln)["id"]: json.loads(ln) for ln in open(root / "meta.jsonl", encoding="utf-8")}
    summary = {}
    for ocr_file in sorted(root.glob("ocr_*.jsonl")):
        engine = ocr_file.stem[4:]
        stats = defaultdict(Counter)
        examples = []
        for ln in open(ocr_file, encoding="utf-8"):
            r = json.loads(ln)
            m = meta[r["image"][:-4]]
            ocr, shown, orig = (normalize_text(x) for x in (r["text"], m["shown"], m["original"]))
            group = "control" if m["kind"] == "control" else "injected"
            if ocr == shown:
                verdict = "faithful"
            elif ocr == orig:
                verdict = "autocorrect"
            else:
                verdict = "other"
            stats[group][verdict] += 1
            stats["kind:" + m["kind"]][verdict] += 1
            if group == "injected" and verdict != "faithful" and len(examples) < 12:
                examples.append({"shown": shown, "original": orig, "ocr": ocr, "verdict": verdict})
        summary[engine] = {k: dict(v) | {"n": sum(v.values())} for k, v in stats.items()}
        summary[engine]["examples"] = examples
        inj = stats["injected"]
        n = sum(inj.values()) or 1
        ctl = stats["control"]
        print(f"[{engine}] injected n={n}: faithful={inj['faithful'] / n:.1%} "
              f"autocorrect={inj['autocorrect'] / n:.1%} other={inj['other'] / n:.1%} | "
              f"control exact={ctl['faithful'] / max(sum(ctl.values()), 1):.1%} (n={sum(ctl.values())})")
        for k in sorted(stats):
            if k.startswith("kind:"):
                c = stats[k]
                print(f"    {k:16} n={sum(c.values()):3} " + " ".join(f"{v}={c[v]}" for v in ("faithful", "autocorrect", "other")))
        for e in examples[:6]:
            print(f"    shown={e['shown']!r} orig={e['original']!r} ocr={e['ocr']!r} -> {e['verdict']}")
    (root / "injection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
