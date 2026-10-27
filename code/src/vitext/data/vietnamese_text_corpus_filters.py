"""Bộ lọc và thống kê cho kho chữ huấn luyện tiếng Việt (mục 6.3 plan.md).

- Bảng 134 ký tự có dấu (67 thường + 67 hoa) dùng để đo độ phủ.
- Lọc ký tự: chỉ chữ Latin, chữ có dấu tiếng Việt, số, dấu câu thông dụng (không có ngoặc kép,
  vì caption đặt chữ đích trong ngoặc kép).
- Lọc "giống tiếng Việt": phần lớn từ có dạng âm tiết tiếng Việt (loại tên Latin, tiếng Anh).
- Khoá mẫu câu (template key) để phát hiện câu bot sinh hàng loạt ("X là một xã thuộc huyện Y").
- Điểm "hiếm" để lấy mẫu dư dấu chồng (ặ ỗ ự ẫ), hỏi/ngã, họ ă và đ.
"""
import re
import string
import unicodedata
from collections import Counter

from vitext.metrics.vietnamese_text_normalization import decompose_char

_BASES = "aăâeêioôơuưy"
_TONES = ["", "́", "̀", "̉", "̃", "̣"]


def _diacritic_chars_lower() -> list:
    chars = {unicodedata.normalize("NFC", b + t) for b in _BASES for t in _TONES}
    chars -= set("aeiouy")  # bỏ nguyên âm trơn
    chars.add("đ")
    return sorted(chars)


VN_DIACRITIC_LOWER = _diacritic_chars_lower()  # 67
VN_DIACRITIC_CHARS = VN_DIACRITIC_LOWER + [c.upper() for c in VN_DIACRITIC_LOWER]  # 134
_PUNCT = " .,;:!?-–()/%&+"
_VN_DIACRITIC_SET = frozenset(VN_DIACRITIC_CHARS)
ALLOWED_CHARS = frozenset(string.ascii_letters + string.digits + _PUNCT) | _VN_DIACRITIC_SET

# Âm tiết tiếng Việt trên chữ cơ sở (đã bỏ dấu, đ -> d): phụ âm đầu? + 1–3 nguyên âm + phụ âm cuối?
_SYLLABLE = re.compile(r"^(ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|[bcdghklmnprstvx])?[aeiouy]{1,3}(ng|nh|ch|[cmnpt])?$")
_NUMBER = re.compile(r"^\d+([.,]\d+)*(%|h)?$")
_WORD_STRIP = ".,;:!?()/%&+-–"


def strip_marks(text: str) -> str:
    """Bỏ mọi dấu, đ -> d; giữ hoa/thường."""
    nfd = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in nfd if not unicodedata.combining(c))


def has_only_allowed_chars(text: str) -> bool:
    return all(c in ALLOWED_CHARS for c in text)


def is_syllable_like(word: str) -> bool:
    w = word.strip(_WORD_STRIP)
    if not w:
        return True  # chỉ là dấu câu
    if _NUMBER.match(w):
        return True
    return len(w) <= 7 and bool(_SYLLABLE.match(strip_marks(w).lower()))


def has_vietnamese_diacritic(word: str) -> bool:
    return any(c in _VN_DIACRITIC_SET for c in word)


def looks_vietnamese(text: str, *, min_syllable_ratio: float = 0.9, min_diacritic_ratio: float = 0.3) -> bool:
    """Đa số từ có dạng âm tiết tiếng Việt và đủ nhiều từ mang dấu."""
    words = text.split()
    if not words:
        return False
    syll = sum(is_syllable_like(w) for w in words) / len(words)
    diac = sum(has_vietnamese_diacritic(w) for w in words) / len(words)
    return syll >= min_syllable_ratio and diac >= min_diacritic_ratio


def template_key(sentence: str) -> str:
    """Thay từ viết hoa/chứa số bằng X để gom các câu cùng khuôn (câu bot sinh)."""
    out = []
    for tok in sentence.split():
        core = tok.strip(_WORD_STRIP)
        is_slot = bool(core) and (core[0].isupper() or any(ch.isdigit() for ch in core))
        tag = "X" if is_slot else tok.lower()
        if not (tag == "X" and out and out[-1] == "X"):
            out.append(tag)
    return " ".join(out)


def mark_features(text: str) -> Counter:
    """Đếm đặc trưng dấu: stacked (dấu phụ + dấu thanh), hoi/nga, trang (họ ă), gach (đ)."""
    feats = Counter()
    for ch in text:
        if ch.isascii():
            continue
        _, diac, tone = decompose_char(ch)
        if diac and tone:
            feats["stacked"] += 1
        if tone in ("hoi", "nga"):
            feats["hoi_nga"] += 1
        if diac == "trang":
            feats["trang"] += 1
        if diac == "gach":
            feats["gach"] += 1
    return feats


def rarity_weight(text: str, *, strength: float = 4.0) -> float:
    """Trọng số lấy mẫu: 1 + strength × mật độ đặc trưng hiếm trên mỗi từ."""
    f = mark_features(text)
    n_words = max(1, len(text.split()))
    score = f["stacked"] + f["hoi_nga"] + 1.5 * f["trang"] + f["gach"]
    return 1.0 + strength * score / n_words


def diacritic_char_counts(texts) -> Counter:
    """Đếm số lần xuất hiện của từng ký tự trong 134 ký tự có dấu."""
    counts = Counter()
    for t in texts:
        counts.update(c for c in t if c in _VN_DIACRITIC_SET)
    return counts


def length_bucket(n_words: int) -> str:
    """Bucket độ dài giống ViGlyph-Bench: short 1–3, medium 4–10, long 11–25 từ."""
    return "short" if n_words <= 3 else "medium" if n_words <= 10 else "long" if n_words <= 25 else "too_long"
