"""Kiểm tra font có đủ 134 ký tự có dấu tiếng Việt (đọc cmap bằng fontTools), ghi danh sách font dùng được.

Usage: python scripts/check-vietnamese-font-coverage.py data/fonts
Ghi <dir>/fonts_ok.json: [{"path", "family", "variable", "instances"}].
"""
import json
import sys
import unicodedata
from pathlib import Path

from fontTools.ttLib import TTFont

BASES = "aăâeêioôơuưy"
TONES = ["", "́", "̀", "̉", "̃", "̣"]


def vietnamese_diacritic_chars() -> list:
    chars = set()
    for b in BASES:
        for t in TONES:
            c = unicodedata.normalize("NFC", b + t)
            if c not in "aeiouy":  # bỏ nguyên âm trơn
                chars.add(c)
    chars.add("đ")
    lower = sorted(chars)
    return lower + [c.upper() for c in lower]


def main():
    font_dir = Path(sys.argv[1])
    need = vietnamese_diacritic_chars()
    print("ký tự có dấu cần có:", len(need))
    ok = []
    for f in sorted(font_dir.glob("*.ttf")):
        font = TTFont(f, lazy=True)
        cmap = font.getBestCmap() or {}
        missing = [c for c in need if ord(c) not in cmap]
        variable = "fvar" in font
        instances = []
        if variable:
            names = font["name"]
            instances = [names.getDebugName(i.subfamilyNameID) for i in font["fvar"].instances]
        family = font["name"].getDebugName(1)
        status = "OK" if not missing else f"THIẾU {len(missing)}: {''.join(missing[:12])}"
        print(f"{f.name:55s} {status}")
        if not missing:
            ok.append({"path": str(f), "family": family, "variable": variable, "instances": instances})
    (font_dir / "fonts_ok.json").write_text(json.dumps(ok, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(ok)} font đủ ký tự -> {font_dir / 'fonts_ok.json'}")


if __name__ == "__main__":
    main()
