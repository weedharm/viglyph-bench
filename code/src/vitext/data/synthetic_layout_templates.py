"""Sáu bố cục ảnh chữ loại A (1024×1024): poster, biển hiệu, bảng phấn, băng rôn, nhãn sản phẩm, bìa sách.

Mỗi hàm nhận (rng, nrng, catalog, texts) với texts[0] là chữ chính, trả về dict
{image, mask, texts, boxes, caption, font} hoặc None nếu chữ chính không vừa khung.
Chữ phụ không vừa thì bỏ, và caption chỉ nhắc những chữ thực sự được vẽ (khớp NFC với texts).
"""
from PIL import Image, ImageDraw, ImageFilter

from vitext.data.synthetic_render_primitives import (
    PALETTE, WEIGHTS, add_noise, contrast, draw_text_block, fit_text, luminance, pick_color, vertical_gradient)

W = H = 1024
LAYOUT_MAX_TEXTS = {"poster": 3, "bien_hieu": 2, "bang_phan": 3, "bang_ron": 1, "nhan_san_pham": 3, "bia_sach": 2}
LONG_TEXT_LAYOUTS = ("poster", "bang_phan")


def _font(rng, cat, texts, category_weights, min_weight="Regular"):
    """Pacifico vẽ chữ hoa trông như chữ thường (V, N, M, Ê) nên không dùng khi có chuỗi viết hoa toàn bộ."""
    exclude = ("Pacifico",) if any(t.isupper() for t in texts) else ()
    return cat.sample(rng, category_weights, min_weight, exclude)


def _style(fc) -> str:
    base = {"sans": "kiểu chữ không chân", "serif": "kiểu chữ có chân", "hand": "kiểu chữ viết tay"}[fc.category]
    return base + (" in đậm" if WEIGHTS.index(fc.weight) >= WEIGHTS.index("Bold") else "")


def _caption(rng, intro, texts, fg, style, long_main, sep=", "):
    """intro kết thúc bằng động từ ("..., in") thì truyền sep=" " để nối liền "in dòng chữ ..."."""
    head = "đoạn văn" if long_main else rng.choice(["dòng chữ", "dòng chữ lớn", "tiêu đề"])
    cap = f"{intro}{sep}{head} \"{texts[0]}\" màu {fg}, {style}"
    if len(texts) > 1:
        cap += rng.choice([", bên dưới là dòng chữ nhỏ hơn \"{}\"", ", phía dưới ghi \"{}\""]).format(texts[1])
    if len(texts) > 2:
        cap += f" và dòng cuối \"{texts[2]}\""
    return cap + "."


def _render_texts(img, mask, texts, fc, slots, long_main, **draw_kw):
    """slots: [(box, max_size, min_size, max_lines)] cho chữ chính và chữ phụ. Trả về (texts đã vẽ, boxes)."""
    done, boxes = [], []
    for i, (text, slot) in enumerate(zip(texts, slots)):
        box, max_size, min_size, max_lines = slot
        if i == 0 and long_main:
            max_size, min_size, max_lines = min(max_size, 96), 36, 9
        fitted = fit_text(text, fc, box[2] - box[0], box[3] - box[1], max_size, min_size, max_lines)
        if fitted is None:
            if i == 0:
                return None, None
            continue
        done.append(text)
        boxes.append(draw_text_block(img, mask, fitted, box, **draw_kw))
    return done, boxes


def _slots_stack(x0, y0, x1, y1, n, main_share=0.6, sizes=((190, 52, 3), (80, 40, 2), (60, 36, 1))):
    """Chia khung dọc: chữ chính chiếm main_share, chữ phụ chia phần còn lại (tối đa len(sizes) dòng)."""
    n = min(n, len(sizes))
    if n == 1:
        return [((x0, y0, x1, y1), *sizes[0])]
    hm = (y1 - y0) * main_share
    rest = (y1 - y0 - hm) / (n - 1)
    slots = [((x0, y0, x1, y0 + hm), *sizes[0])]
    for k in range(1, n):
        top = y0 + hm + (k - 1) * rest
        slots.append(((x0 + 30, top + 10, x1 - 30, top + rest), *sizes[k]))
    return slots


def _result(img, mask, texts, boxes, caption, fc):
    return {"image": img, "mask": mask, "texts": texts, "boxes": boxes, "caption": caption, "font": fc.label}


def poster(rng, nrng, cat, texts, long_main=False):
    bg_name, bg = rng.choice(list(PALETTE.items()))
    fg_name, fg = pick_color(rng, bg, 4.5)
    if rng.random() < 0.5:
        img, bg_desc = Image.new("RGB", (W, H), bg), bg_name
    else:  # chuyển sắc sang màu gần nền
        c2 = tuple(int(v * 0.78) if luminance(bg) > 0.4 else int(v + (255 - v) * 0.3) for v in bg)
        img, bg_desc = vertical_gradient((W, H), bg, c2), f"{bg_name} chuyển sắc"
    d = ImageDraw.Draw(img)
    for _ in range(rng.randint(0, 3)):  # hình trang trí mờ ở góc
        cx, cy, r = rng.choice([0, W]), rng.choice([0, H]), rng.randint(90, 180)
        tint = tuple(int(0.8 * a + 0.2 * b) for a, b in zip(bg, PALETTE[rng.choice(list(PALETTE))]))
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=tint)
    fc = _font(rng, cat, texts, {"sans": 0.65, "serif": 0.25, "hand": 0.10}, "Medium")
    mask = Image.new("L", (W, H), 0)
    slots = _slots_stack(90, 150, W - 90, H - 120, len(texts), 0.62 if not long_main else 0.78)
    done, boxes = _render_texts(img, mask, texts, fc, slots, long_main, fill=fg, align="center")
    if done is None:
        return None
    intro = rng.choice([f"Một tấm poster thiết kế phẳng nền {bg_desc}", f"Poster đồ hoạ tối giản trên nền {bg_desc}"])
    return _result(img, mask, done, boxes, _caption(rng, intro, done, fg_name, _style(fc), long_main), fc)


def _brick_wall(nrng):
    img = Image.new("RGB", (W, H), (150, 78, 58))
    d = ImageDraw.Draw(img)
    for row, y in enumerate(range(0, H, 44)):
        d.line((0, y, W, y), fill=(190, 182, 170), width=5)
        for x in range(-60 if row % 2 else 0, W, 120):
            d.line((x, y, x, y + 44), fill=(190, 182, 170), width=5)
    return add_noise(img, nrng, 10)


def bien_hieu(rng, nrng, cat, texts, long_main=False):
    if rng.random() < 0.5:
        img, wall = _brick_wall(nrng), "tường gạch đỏ"
    else:
        wname, wc = rng.choice([("trắng", PALETTE["trắng"]), ("vàng nhạt", PALETTE["vàng nhạt"]), ("xám nhạt", PALETTE["xám nhạt"])])
        img, wall = add_noise(Image.new("RGB", (W, H), wc), nrng, 6), f"tường vôi {wname}"
    board_name, board = rng.choice(list(PALETTE.items()))
    border_name, border = pick_color(rng, board, 1.8)
    fg_name, fg = pick_color(rng, board, 4.5, exclude=(border_name,))
    bh = rng.randint(300, 460)
    x0, y0 = rng.randint(40, 90), 512 - bh // 2 + rng.randint(-60, 60)
    x1, y1 = W - x0, y0 + bh
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((x0 + 10, y0 + 12, x1 + 10, y1 + 12), rng.randint(0, 24), fill=(40, 40, 40))  # bóng đổ
    d.rounded_rectangle((x0, y0, x1, y1), rng.randint(0, 24), fill=board, outline=border, width=rng.randint(10, 18))
    fc = _font(rng, cat, texts, {"sans": 0.8, "serif": 0.2}, "SemiBold")
    mask = Image.new("L", (W, H), 0)
    slots = _slots_stack(x0 + 45, y0 + 35, x1 - 45, y1 - 35, len(texts), 0.68, ((200, 52, 2), (70, 38, 1)))
    done, boxes = _render_texts(img, mask, texts, fc, slots, long_main, fill=fg, align="center")
    if done is None:
        return None
    intro = rng.choice([f"Biển hiệu cửa hàng nền {board_name} viền {border_name}, gắn trên {wall}",
                        f"Tấm biển nền {board_name} có viền {border_name} treo trên {wall}"])
    return _result(img, mask, done, boxes, _caption(rng, intro, done, fg_name, _style(fc), long_main), fc)


def bang_phan(rng, nrng, cat, texts, long_main=False):
    board_name, board = rng.choice([("xanh", (34, 60, 46)), ("đen", (30, 33, 32))])
    img = Image.new("RGB", (W, H), (120, 78, 44))  # khung gỗ
    ImageDraw.Draw(img).rectangle((36, 36, W - 36, H - 36), fill=board)
    smudge = Image.new("L", (W, H), 0)
    sd = ImageDraw.Draw(smudge)
    for _ in range(rng.randint(3, 8)):  # vết phấn lau dở
        x, y = rng.randint(60, W - 160), rng.randint(60, H - 120)
        sd.ellipse((x, y, x + rng.randint(80, 260), y + rng.randint(30, 90)), fill=rng.randint(10, 26))
    img.paste((230, 230, 225), (0, 0), smudge.filter(ImageFilter.GaussianBlur(25)))
    img = add_noise(img, nrng, 7)
    fg_name, fg = rng.choices([("trắng", (238, 238, 232)), ("vàng nhạt", PALETTE["vàng nhạt"]),
                               ("hồng pastel", PALETTE["hồng pastel"]), ("xanh pastel", PALETTE["xanh pastel"])],
                              weights=[0.6, 0.15, 0.125, 0.125])[0]
    fc = _font(rng, cat, texts, {"hand": 0.7, "sans": 0.3})
    mask = Image.new("L", (W, H), 0)
    slots = _slots_stack(100, 120, W - 100, H - 110, len(texts), 0.6 if not long_main else 0.8)
    align = "left" if long_main and rng.random() < 0.5 else "center"
    done, boxes = _render_texts(img, mask, texts, fc, slots, long_main, fill=fg, align=align, chalk_rng=nrng)
    if done is None:
        return None
    intro = rng.choice([f"Bảng phấn {board_name} khung gỗ trong quán cà phê", f"Tấm bảng {board_name} viền gỗ treo tường"])
    cap = _caption(rng, intro + ", viết bằng phấn", done, fg_name, "nét phấn " + _style(fc), long_main, " ")
    return _result(img, mask, done, boxes, cap, fc)


def bang_ron(rng, nrng, cat, texts, long_main=False):
    if rng.random() < 0.6:
        img, scene = vertical_gradient((W, H), PALETTE["xanh pastel"], (250, 250, 252)), "bầu trời xanh"
    else:
        wname, wc = rng.choice([("be", PALETTE["be"]), ("xám nhạt", PALETTE["xám nhạt"]), ("kem", PALETTE["kem"])])
        img, scene = Image.new("RGB", (W, H), wc), f"bức tường màu {wname}"
    band_name, band = ("đỏ", PALETTE["đỏ"]) if rng.random() < 0.6 else rng.choice(list(PALETTE.items()))
    fg_name, fg = ("vàng", PALETTE["vàng"]) if band_name == "đỏ" and rng.random() < 0.7 else pick_color(rng, band, 4.5)
    bh = rng.randint(200, 300)
    y0 = 512 - bh // 2 + rng.randint(-80, 80)
    d = ImageDraw.Draw(img)
    for x in (18, W - 18):  # dây buộc hai đầu
        d.line((x, y0 - 120, x, y0 + bh + 120), fill=(90, 90, 90), width=6)
    d.rectangle((18, y0, W - 18, y0 + bh), fill=band)
    img = add_noise(img, nrng, 4)
    fc = _font(rng, cat, texts, {"sans": 0.85, "serif": 0.15}, "Bold")
    mask = Image.new("L", (W, H), 0)
    slots = [((60, y0 + 22, W - 60, y0 + bh - 22), 170, 44, 2)]
    done, boxes = _render_texts(img, mask, texts[:1], fc, slots, long_main, fill=fg, align="center")
    if done is None:
        return None
    intro = f"Một băng rôn vải {band_name} căng ngang trước {scene}, in"
    return _result(img, mask, done, boxes, _caption(rng, intro, done, fg_name, _style(fc), long_main, " "), fc)


def nhan_san_pham(rng, nrng, cat, texts, long_main=False):
    bg_name, bg = rng.choice([(n, PALETTE[n]) for n in ("trắng", "kem", "be", "xám nhạt", "xanh pastel")])
    img = Image.new("RGB", (W, H), bg)
    label_name, label = rng.choice([(n, c) for n, c in PALETTE.items() if contrast(c, bg) > 1.3])
    fg_name, fg = pick_color(rng, label, 4.5)
    lw, lh = rng.randint(600, 780), rng.randint(600, 760)
    x0, y0 = (W - lw) // 2, (H - lh) // 2
    sh = Image.new("L", (W, H), 0)
    ImageDraw.Draw(sh).rounded_rectangle((x0 + 14, y0 + 18, x0 + lw + 14, y0 + lh + 18), 40, fill=110)
    img.paste((30, 30, 30), (0, 0), sh.filter(ImageFilter.GaussianBlur(14)))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((x0, y0, x0 + lw, y0 + lh), rng.randint(20, 60), fill=label)
    if rng.random() < 0.5:
        d.rounded_rectangle((x0 + 22, y0 + 22, x0 + lw - 22, y0 + lh - 22), 30, outline=fg, width=4)
    fc = _font(rng, cat, texts, {"sans": 0.6, "serif": 0.3, "hand": 0.1}, "Medium")
    mask = Image.new("L", (W, H), 0)
    slots = _slots_stack(x0 + 55, y0 + 60, x0 + lw - 55, y0 + lh - 60, len(texts), 0.6,
                         ((150, 48, 3), (70, 38, 2), (56, 36, 1)))
    done, boxes = _render_texts(img, mask, texts, fc, slots, long_main, fill=fg, align="center")
    if done is None:
        return None
    intro = f"Nhãn sản phẩm bo góc màu {label_name} đặt trên nền {bg_name}, in"
    return _result(img, mask, done, boxes, _caption(rng, intro, done, fg_name, _style(fc), long_main, " "), fc)


def bia_sach(rng, nrng, cat, texts, long_main=False):
    bg_name, bg = rng.choice([(n, PALETTE[n]) for n in ("trắng", "kem", "be", "xám nhạt")])
    img = Image.new("RGB", (W, H), bg)
    cover_name, cover = rng.choice([(n, c) for n, c in PALETTE.items() if contrast(c, bg) > 1.3])
    fg_name, fg = pick_color(rng, cover, 4.5)
    bw, bh = rng.randint(600, 680), rng.randint(860, 920)
    x0, y0 = (W - bw) // 2, (H - bh) // 2
    sh = Image.new("L", (W, H), 0)
    ImageDraw.Draw(sh).rectangle((x0 + 12, y0 + 16, x0 + bw + 12, y0 + bh + 16), fill=120)
    img.paste((30, 30, 30), (0, 0), sh.filter(ImageFilter.GaussianBlur(12)))
    d = ImageDraw.Draw(img)
    d.rectangle((x0, y0, x0 + bw, y0 + bh), fill=cover)
    d.rectangle((x0, y0, x0 + 18, y0 + bh), fill=tuple(int(v * 0.8) for v in cover))  # gáy sách
    ly = y0 + int(bh * 0.66)
    d.line((x0 + 80, ly, x0 + bw - 80, ly), fill=fg, width=3)
    fc = _font(rng, cat, texts, {"serif": 0.55, "sans": 0.35, "hand": 0.10}, "Medium")
    mask = Image.new("L", (W, H), 0)
    slots = [((x0 + 60, y0 + 90, x0 + bw - 50, ly - 40), 140, 44, 4),
             ((x0 + 70, ly + 40, x0 + bw - 60, y0 + bh - 60), 64, 36, 2)]
    done, boxes = _render_texts(img, mask, texts, fc, slots, long_main, fill=fg, align="center")
    if done is None:
        return None
    intro = f"Bìa sách nền {cover_name}, in"
    cap = f"{intro} tiêu đề \"{done[0]}\" màu {fg_name}, {_style(fc)}"
    if len(done) > 1:
        cap += f", tên tác giả \"{done[1]}\" ở phía dưới"
    return _result(img, mask, done, boxes, cap + ".", fc)


LAYOUTS = {"poster": poster, "bien_hieu": bien_hieu, "bang_phan": bang_phan, "bang_ron": bang_ron,
           "nhan_san_pham": nhan_san_pham, "bia_sach": bia_sach}
