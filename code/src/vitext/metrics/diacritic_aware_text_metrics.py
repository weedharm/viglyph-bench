"""Metric nhạy dấu cho chữ tiếng Việt (mục 6.1 plan.md).

Căn chỉnh hai tầng: Levenshtein trên ký tự NFC (thay thế cùng chữ cơ sở rẻ hơn),
rồi so từng cặp ký tự theo (chữ cơ sở, dấu phụ, dấu thanh) để phân rã lỗi không mơ hồ.

OCR có thể đọc thêm chữ khác trong ảnh, nên mặc định căn chỉnh semi-global:
chuỗi đích phải khớp hết, phần thừa đầu/cuối của OCR không bị phạt (span được nới
tới ranh giới từ). `exact_full` vẫn báo khớp toàn chuỗi theo đúng định nghĩa plan.
"""
import re
from collections import Counter

from .vietnamese_text_normalization import (
    DIACRITIC_MARKS, TONE_MARKS, decompose_char, is_letter_or_digit, normalize_text, to_nfd_mapped,
)

_INDEL = 5  # chi phí nguyên (x5) để tránh so sánh số thực khi backtrace


def _weighted_sub(a: str, b: str) -> int:
    if a == b:
        return 0
    base_a, base_b = decompose_char(a)[0], decompose_char(b)[0]
    if base_a == base_b:
        return 3  # chỉ khác dấu
    return 4 if base_a.lower() == base_b.lower() else 5


def _unit_sub(a: str, b: str) -> int:
    return 0 if a == b else _INDEL


def align(target: str, pred: str, sub_cost=_weighted_sub, semi_global: bool = False):
    """Trả về (distance, span_start, span_end, ops); ops = [(op, t_char, p_char)]."""
    n, m = len(target), len(pred)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * _INDEL
    for j in range(1, m + 1):
        dp[0][j] = 0 if semi_global else j * _INDEL
    for i in range(1, n + 1):
        t_char, row, prev = target[i - 1], dp[i], dp[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev[j - 1] + sub_cost(t_char, pred[j - 1]), prev[j] + _INDEL, row[j - 1] + _INDEL)
    end = min(range(m + 1), key=lambda j: dp[n][j]) if semi_global else m
    ops, i, j = [], n, end
    while i > 0 or (j > 0 and not semi_global):
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + sub_cost(target[i - 1], pred[j - 1]):
            op = "match" if target[i - 1] == pred[j - 1] else "sub"
            ops.append((op, target[i - 1], pred[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + _INDEL:
            ops.append(("del", target[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, pred[j - 1]))
            j -= 1
    ops.reverse()
    return dp[n][end], j, end, ops


def edit_distance(a: str, b: str) -> int:
    return align(a, b, sub_cost=_unit_sub)[0] // _INDEL


def _extend_to_word_boundary(text: str, start: int, end: int) -> tuple:
    while start > 0 and is_letter_or_digit(text[start - 1]):
        start -= 1
    while end < len(text) and is_letter_or_digit(text[end]):
        end += 1
    return start, end


def _char_errors(char: str) -> Counter:
    """Lỗi khi cả ký tự bị thiếu/thừa: chữ cơ sở + từng dấu (khớp cách đếm của CER_NFD)."""
    base, diacritic, tone = decompose_char(char)
    if not base.isalnum():
        return Counter(other=1)
    return Counter(base=1, diacritic=int(diacritic is not None), tone=int(tone is not None))


def decompose_errors(ops) -> dict:
    errors = Counter(base=0, tone=0, diacritic=0, other=0)
    tone_total, tone_ok, diac_total, diac_ok = Counter(), Counter(), Counter(), Counter()
    for op, t_char, p_char in ops:
        if t_char is not None:
            _, t_diac, t_tone = decompose_char(t_char)
            p_diac, p_tone = decompose_char(p_char)[1:] if p_char is not None else (None, None)
            if t_tone:
                tone_total[t_tone] += 1
                tone_ok[t_tone] += int(p_tone == t_tone)
            if t_diac:
                diac_total[t_diac] += 1
                diac_ok[t_diac] += int(p_diac == t_diac)
        if op == "del":
            errors.update(_char_errors(t_char))
        elif op == "ins":
            errors.update(_char_errors(p_char))
        elif op == "sub":
            (tb, td, tt), (pb, pd, pt) = decompose_char(t_char), decompose_char(p_char)
            if tb != pb:
                errors["base" if tb.isalnum() else "other"] += 1
            errors["diacritic"] += int(td != pd)
            errors["tone"] += int(tt != pt)
    return {
        "errors": dict(errors),
        "tone_total": dict(tone_total), "tone_ok": dict(tone_ok),
        "diac_total": dict(diac_total), "diac_ok": dict(diac_ok),
    }


def _contains_word(haystack: str, needle: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", haystack) is not None


def score_pair(target: str, pred: str, semi_global: bool = True) -> dict:
    """Chấm một cặp (chuỗi đích, chuỗi OCR)."""
    t, p_full = normalize_text(target), normalize_text(pred)
    if semi_global and p_full:
        _, start, end, _ = align(t, p_full, semi_global=True)
        start, end = _extend_to_word_boundary(p_full, start, end)
        p = p_full[start:end]
    else:
        p = p_full
    d_nfc = edit_distance(t, p)
    t_nfd, p_nfd = to_nfd_mapped(t), to_nfd_mapped(p)
    d_nfd = edit_distance(t_nfd, p_nfd)
    result = {
        "target_norm": t, "pred_norm": p_full, "pred_span": p,
        "exact_full": t == p_full,
        "exact_full_ci": t.casefold() == p_full.casefold(),
        "exact_span": _contains_word(p_full, t),
        "exact_span_ci": _contains_word(p_full.casefold(), t.casefold()),
        "ned": 1.0 - d_nfc / max(len(t), len(p), 1),
        "cer_nfc": d_nfc / max(len(t), 1),
        "cer_nfd": d_nfd / max(len(t_nfd), 1),
        "n_chars_nfc": len(t), "n_chars_nfd": len(t_nfd),
    }
    result.update(decompose_errors(align(t, p)[3]))
    return result


TONE_NAMES = list(TONE_MARKS.values())
DIACRITIC_NAMES = list(DIACRITIC_MARKS.values())
