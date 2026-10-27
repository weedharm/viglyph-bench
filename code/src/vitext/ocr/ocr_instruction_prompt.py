"""Câu lệnh OCR dùng chung cho mọi engine VLM: chép nguyên văn, cấm tự sửa dấu/chính tả."""

OCR_INSTRUCTION = (
    "Hãy chép lại chính xác toàn bộ chữ xuất hiện trong ảnh, giữ nguyên dấu tiếng Việt, "
    "chính tả và chữ hoa/thường đúng như trong ảnh. Nếu chữ trong ảnh bị sai dấu hoặc sai chính tả "
    "thì vẫn chép đúng như ảnh, không được sửa. Không giải thích, chỉ trả về văn bản."
)
