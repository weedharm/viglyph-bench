"""Chấm kết quả trang chấm mù: giải mã trái/phải bằng key.json, báo tỉ lệ đúng hoàn toàn của từng model,
tỉ lệ sai dấu, ưa thích, và kiểm định McNemar trên cặp (A đúng/B sai so với A sai/B đúng).

Usage: python scripts/score-blind-human-eval.py outputs/human_eval_v1_key.json human_eval_v1_ket_qua.json
"""
import json
import sys
from collections import Counter
from math import comb


def p_two_sided(k, n):
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n) if n else 1.0


def main():
    key = json.load(open(sys.argv[1], encoding="utf-8"))
    ans = json.load(open(sys.argv[2], encoding="utf-8"))
    per = {"A": Counter(), "B": Counter()}
    pref, a_only, b_only, n = Counter(), 0, 0, 0
    buckets = {}
    for item_id, meta in key["items"].items():
        a = ans.get(item_id, {})
        if not (a.get("l") and a.get("r")):
            continue
        n += 1
        lab = {meta["l"]: a["l"], meta["r"]: a["r"]}
        for m in ("A", "B"):
            per[m][lab[m]] += 1
            buckets.setdefault((meta["bucket"], m), Counter())[lab[m]] += 1
        a_only += lab["A"] == "ok" and lab["B"] != "ok"
        b_only += lab["B"] == "ok" and lab["A"] != "ok"
        if a.get("pref"):
            pref[{"l": meta["l"], "r": meta["r"], "eq": "="}[a["pref"]]] += 1
    print(f"A = {key['A']}\nB = {key['B']}\n{n} cặp đã chấm")
    for m in ("A", "B"):
        c = per[m]
        print(f"  {m}: đúng hoàn toàn {c['ok'] / n:.3f}  sai dấu {c['diac'] / n:.3f}  sai/thiếu chữ {c['char'] / n:.3f}")
    for b in ("short", "medium", "long"):
        row = [f"{m} {buckets[(b, m)]['ok'] / max(1, sum(buckets[(b, m)].values())):.3f}" for m in ("A", "B") if (b, m) in buckets]
        print(f"  {b}: " + "  ".join(row))
    print(f"  Chỉ A đúng {a_only}, chỉ B đúng {b_only}, McNemar p = {p_two_sided(min(a_only, b_only), a_only + b_only):.4f}")
    print(f"  Ưa thích: A {pref['A']}, B {pref['B']}, như nhau {pref['=']}")


if __name__ == "__main__":
    main()
