"""Chuẩn hoá chuỗi tiếng Việt và tách ký tự thành (chữ cơ sở, dấu phụ, dấu thanh).

Hợp đồng theo mục 6.1 của plan.md:
- So sánh ở dạng NFC; tách dấu ở dạng NFD.
- đ/Đ không tách được bằng NFD nên ánh xạ đ -> d + U+E000, Đ -> D + U+E000.
- Thống nhất kiểu bỏ dấu cũ/mới (hòa/hoà, khỏe/khoẻ, thủy/thuỷ) để không tính nhầm lỗi.
"""
import re
import unicodedata
from functools import lru_cache

TONE_MARKS = {
    "́": "sac",
    "̀": "huyen",
    "̉": "hoi",
    "̃": "nga",
    "̣": "nang",
}
DBAR = ""  # ký tự vùng riêng, đóng vai "dấu gạch" của đ/Đ
DIACRITIC_MARKS = {
    "̂": "mu",  # â ê ô
    "̛": "moc",  # ơ ư
    "̆": "trang",  # ă
    DBAR: "gach",  # đ
}

# Nháy cong, gạch dài, khoảng trắng không ngắt -> dạng ASCII để OCR không bị phạt oan.
_QUOTE_DASH_MAP = str.maketrans({
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "‘": "'", "’": "'",
    "–": "-", "—": "-", "−": "-",
    " ": " ",
})

_TONE_CLASS = "[" + "".join(TONE_MARKS) + "]"
# Kiểu cũ đặt dấu ở nguyên âm đầu của âm tiết mở "oa/oe/uy" (hòa, khỏe, thủy).
# Chuyển về kiểu mới (hoà, khoẻ, thuỷ). Chạy trên chuỗi NFD; lookahead đảm bảo âm tiết kết thúc.
# Chỉ o+(a|e) và u+y: vần "ua" (của, mùa, lúa) và "ue" luôn đặt dấu trên u ở cả hai kiểu.
_OLD_STYLE_OA_OE = re.compile(r"([oO])(" + _TONE_CLASS + r")([aAeE])(?![^\W\d_])")
_OLD_STYLE_UY = re.compile(r"([uU])(" + _TONE_CLASS + r")([yY])(?![^\W\d_])")


def unify_tone_placement(text: str) -> str:
    """Đưa vị trí dấu thanh về kiểu mới (hoà, khoẻ, thuỷ). Trả về NFC."""
    nfd = unicodedata.normalize("NFD", text)
    nfd = _OLD_STYLE_UY.sub(r"\1\3\2", _OLD_STYLE_OA_OE.sub(r"\1\3\2", nfd))
    return unicodedata.normalize("NFC", nfd)


def normalize_text(text: str, *, unify_tone_style: bool = True, casefold: bool = False) -> str:
    """NFC + thống nhất nháy/gạch + gộp khoảng trắng + bỏ ngoặc kép bao ngoài."""
    s = unicodedata.normalize("NFC", text or "").translate(_QUOTE_DASH_MAP)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip("\"'").strip()
    if unify_tone_style:
        s = unify_tone_placement(s)
    if casefold:
        s = s.casefold()
    return s


def to_nfd_mapped(text: str) -> str:
    """NFD với đ/Đ đã ánh xạ sang d/D + DBAR; dùng cho CER_NFD."""
    return unicodedata.normalize("NFD", text).replace("đ", "d" + DBAR).replace("Đ", "D" + DBAR)


@lru_cache(maxsize=4096)
def decompose_char(char: str) -> tuple:
    """Tách một ký tự NFC thành (chữ cơ sở, dấu phụ | None, dấu thanh | None)."""
    nfd = to_nfd_mapped(char)
    base, diacritic, tone = nfd[0], None, None
    for mark in nfd[1:]:
        if mark in TONE_MARKS:
            tone = TONE_MARKS[mark]
        elif mark in DIACRITIC_MARKS:
            diacritic = DIACRITIC_MARKS[mark]
    return base, diacritic, tone


def is_letter_or_digit(char: str) -> bool:
    return decompose_char(char)[0].isalnum()
