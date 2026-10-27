"""So OCR trên ảnh gốc vs ảnh qua VAE (encode->decode) theo cỡ chữ.

Nếu tỉ lệ đúng giảm rõ sau VAE ở cỡ chữ mục tiêu -> VAE làm mất dấu (cổng G1 plan.md).
Usage: python score-vae-roundtrip-ocr.py outputs/vae_roundtrip
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vitext.metrics.diacritic_aware_text_metrics import score_pair  # noqa: E402


def main():
    root = Path(sys.argv[1])
    meta = {json.loads(ln)["image"]: json.loads(ln) for ln in open(root / "meta.jsonl", encoding="utf-8")}
    table = defaultdict(lambda: defaultdict(list))
    for variant in ("orig", "recon"):
        for f in sorted((root / f"{variant}_ocr").glob("ocr_*.jsonl")):
            engine = f.stem[4:]
            for ln in open(f, encoding="utf-8"):
                o = json.loads(ln)
                m = meta[o["image"]]
                table[(engine, m["font_px"])][variant].append(score_pair(m["text"], o["text"]))
    summary = []
    print("| OCR | font px | n | exact gốc | exact sau VAE | CER_NFD gốc | CER_NFD sau VAE | PSNR dB |")
    print("|---|---|---|---|---|---|---|---|")
    for (engine, px), d in sorted(table.items()):
        o, r = d["orig"], d["recon"]
        if not o or not r:
            continue
        ps = [m["psnr"] for m in meta.values() if m["font_px"] == px]
        row = {"engine": engine, "font_px": px, "n": len(o),
               "exact_orig": sum(s["exact_span"] for s in o) / len(o),
               "exact_recon": sum(s["exact_span"] for s in r) / len(r),
               "cer_nfd_orig": sum(s["cer_nfd"] for s in o) / len(o),
               "cer_nfd_recon": sum(s["cer_nfd"] for s in r) / len(r),
               "psnr": sum(ps) / len(ps)}
        summary.append(row)
        print(f"| {engine} | {px} | {row['n']} | {row['exact_orig']:.2f} | {row['exact_recon']:.2f} | "
              f"{row['cer_nfd_orig']:.3f} | {row['cer_nfd_recon']:.3f} | {row['psnr']:.1f} |")
    (root / "vae_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
