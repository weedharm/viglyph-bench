"""Dựng ViGlyph-Bench với caption tiếng Anh theo đúng định dạng prompt của Z-LAN (PR Letters 2026):
"zlan-Vietnamese, <cảnh bằng tiếng Anh> with the Vietnamese text '<chữ đích>' ...". Giữ nguyên id, seed, chữ đích, bucket.
Mỗi template tiếng Việt của ViGlyph-Bench được dịch sát nghĩa sang tiếng Anh.

Usage: python scripts/build-english-template-prompts-for-zlan.py --src bench/viglyph/prompts_v1.jsonl \
           --out bench/viglyph/prompts_v1_en_zlan.jsonl [--no-trigger]
"""
import argparse
import json

TEMPLATES = {
    "sign_0": "a photo of a shop signboard on an old-quarter street facade, with the Vietnamese text '{t}' written in large, clear letters",
    "sign_1": "a modern flat-design poster with a pastel background and the large Vietnamese text '{t}' in the middle",
    "sign_2": "a black chalkboard in a coffee shop with the Vietnamese text '{t}' handwritten in white chalk",
    "sign_3": "a glowing neon sign on a brick wall at night, displaying the Vietnamese text '{t}'",
    "sign_4": "a minimalist book cover with the bold Vietnamese title '{t}' in the center",
    "sign_5": "a red cloth banner stretched across the street, printed with the yellow Vietnamese text '{t}'",
    "sign_6": "a kraft paper card on a wooden table, printed with the Vietnamese text '{t}' in black ink",
    "sign_7": "a product box on a white background, with the Vietnamese text '{t}' printed on the front",
    "long_0": "a large poster on a wall, printed with the Vietnamese passage '{t}' in a clear, legible font",
    "long_1": "a white sheet of paper on a wooden table, printed with the Vietnamese passage '{t}' in crisp black letters",
    "long_2": "a white notice board hanging in a lobby, printed with the Vietnamese text '{t}'",
    "long_3": "a dark blue presentation slide with the white Vietnamese text '{t}'",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-trigger", action="store_true", help="bỏ tiền tố 'zlan-Vietnamese, ' (cho Z-Image gốc)")
    args = ap.parse_args()
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for ln in open(args.src, encoding="utf-8"):
            r = json.loads(ln)
            body = TEMPLATES[r["template"]].format(t=r["text"]) + "."
            r["prompt"] = body[0].upper() + body[1:] if args.no_trigger else "zlan-Vietnamese, " + body
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(n, "prompts →", args.out)


if __name__ == "__main__":
    main()
