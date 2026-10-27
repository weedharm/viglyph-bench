"""Trang chấm mù cho người: mỗi prompt hai ảnh (model A, model B, cùng seed), xếp trái/phải ngẫu nhiên, không tên model,
không kết quả OCR. Người chấm đánh giá từng ảnh (đúng hoàn toàn / sai dấu / sai chữ) và chọn ảnh tốt hơn;
kết quả lưu trong trình duyệt và tải về thành JSON. Khoá ánh xạ (key.json) để riêng, KHÔNG gửi cho người chấm.

Usage: python scripts/build-blind-human-eval-pairs.py --prompts bench/viglyph/prompts_draft_v0.jsonl \
           --a outputs/baseline_1024 --b outputs/eval_rl_nft_v3b_ckpt-80 --n 100 --out outputs/human_eval_v1
"""
import argparse
import html
import json
import random
from pathlib import Path

from PIL import Image

SHARE = {"short": 0.4, "medium": 0.4, "long": 0.2}
PAGE = """<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chấm chữ có dấu</title><style>
body{{font-family:system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}}
.item{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;margin:0 0 18px}}
.target{{font-size:20px;margin:4px 0 10px}} .pair{{display:flex;gap:12px;flex-wrap:wrap}}
.cell{{flex:1 1 320px}} img{{width:100%;max-width:480px;border:1px solid #ccc}}
label{{margin-right:10px;white-space:nowrap}} .bar{{position:sticky;top:0;background:#fafafa;padding:8px 0;z-index:1}}
</style></head><body>
<div class="bar"><b>Chấm chữ tiếng Việt có dấu</b> — đã chấm <span id="done">0</span>/{n}.
<button onclick="dl()">Tải kết quả (JSON)</button></div>
<p>Với mỗi ảnh: chữ trong ảnh có <b>khớp hoàn toàn</b> chữ đích (kể cả dấu thanh, dấu mũ/móc/trăng, chữ đ) không?
Bỏ qua khác biệt hoa/thường và xuống dòng. Sau đó chọn ảnh có chữ đúng hơn.</p>
{items}
<script>
const KEY="human_eval_v1"; let st={{}}; try{{st=JSON.parse(localStorage.getItem(KEY)||"{{}}")}}catch(e){{}}
function save(){{try{{localStorage.setItem(KEY,JSON.stringify(st))}}catch(e){{}};
 document.getElementById("done").textContent=Object.values(st).filter(v=>v.l&&v.r&&v.pref).length}}
document.querySelectorAll("input[type=radio]").forEach(el=>{{const [id,f]=el.name.split("|");
 if(st[id]&&st[id][f]===el.value)el.checked=true;
 el.addEventListener("change",()=>{{st[id]=st[id]||{{}};st[id][f]=el.value;save()}})}});save();
function dl(){{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(st,null,1)],{{type:"application/json"}}));
 a.download="human_eval_v1_ket_qua.json";a.click()}}
</script></body></html>"""


def radios(item_id, field, options):
    return "".join(f'<label><input type="radio" name="{item_id}|{field}" value="{v}"> {t}</label>' for v, t in options)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    prompts = [json.loads(ln) for ln in open(args.prompts, encoding="utf-8")]
    picked = []
    for b, s in SHARE.items():
        pool = [p for p in prompts if p["bucket"] == b]
        picked += rng.sample(pool, min(len(pool), round(args.n * s)))
    rng.shuffle(picked)
    out = Path(args.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    opts = [("ok", "Đúng hoàn toàn"), ("diac", "Sai dấu"), ("char", "Sai/thiếu chữ")]
    key, items = {}, []
    for k, p in enumerate(picked):
        fname = f"{p['id']}_s{p['seed']}.png"
        sides = [("A", Path(args.a) / "images" / fname), ("B", Path(args.b) / "images" / fname)]
        rng.shuffle(sides)
        item_id = f"q{k:03d}"
        key[item_id] = {"prompt_id": p["id"], "text": p["text"], "bucket": p["bucket"], "l": sides[0][0], "r": sides[1][0]}
        cells = []
        for side, (_, src) in zip(("l", "r"), sides):
            rel = f"img/{item_id}_{side}.jpg"
            Image.open(src).convert("RGB").resize((768, 768)).save(out / rel, quality=88)
            cells.append(f'<div class="cell"><img src="{rel}" loading="lazy"><br>'
                         f'Ảnh {"1" if side == "l" else "2"}: {radios(item_id, side, opts)}</div>')
        pref = radios(item_id, "pref", [("l", "Ảnh 1 tốt hơn"), ("r", "Ảnh 2 tốt hơn"), ("eq", "Như nhau")])
        items.append(f'<div class="item"><div>#{k + 1}</div><div class="target">Chữ đích: <b>{html.escape(p["text"])}</b></div>'
                     f'<div class="pair">{"".join(cells)}</div><p>{pref}</p></div>')
    (out / "index.html").write_text(PAGE.format(n=len(picked), items="\n".join(items)), encoding="utf-8")
    key_path = out.parent / f"{out.name}_key.json"  # để ngoài thư mục gửi người chấm
    key_path.write_text(json.dumps({"A": str(args.a), "B": str(args.b), "items": key}, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
    print(f"{len(picked)} cặp → {out}/index.html (khoá: {key_path} — không gửi cho người chấm)")


if __name__ == "__main__":
    main()
