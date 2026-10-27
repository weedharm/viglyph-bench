"""Dataset đọc cache do scripts/cache-latents-and-prompt-embeds.py tạo, và collate đệm text embedding.

Text embedding có độ dài khác nhau: đệm phải bằng 0 và trả mask bool (B, L) như pipeline;
nếu cả lô cùng độ dài thì mask = None (pipeline cũng bỏ mask khi không có đệm).
"""
import json
from pathlib import Path

import torch


class CachedLatentDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir, ids=None):
        self.dir = Path(cache_dir)
        if ids is None:
            ids = [json.loads(ln)["id"] for ln in open(self.dir / "index.jsonl", encoding="utf-8")]
        self.ids = sorted(set(ids))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        d = torch.load(self.dir / f"{self.ids[i]}.pt", map_location="cpu", weights_only=True)
        return d["latents"], d["prompt_embeds"], d["token_mask"]


def collate_cached(batch):
    latents = torch.stack([b[0] for b in batch])
    token_mask = torch.stack([b[2] for b in batch])
    lens = [b[1].shape[0] for b in batch]
    max_len, dim = max(lens), batch[0][1].shape[1]
    embeds = batch[0][1].new_zeros(len(batch), max_len, dim)
    emb_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, b in enumerate(batch):
        embeds[i, : lens[i]] = b[1]
        emb_mask[i, : lens[i]] = True
    return {"latents": latents, "prompt_embeds": embeds, "prompt_mask": None if emb_mask.all() else emb_mask,
            "token_mask": token_mask}


def infinite_loader(dataset, batch_size, seed, num_workers=4):
    """Lặp vô hạn, xáo trộn lại mỗi epoch theo seed + epoch (tái lập được khi resume)."""
    epoch = 0
    while True:
        g = torch.Generator().manual_seed(seed + epoch)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g,
                                             drop_last=True, num_workers=num_workers, collate_fn=collate_cached,
                                             persistent_workers=False, pin_memory=True)
        yield from loader
        epoch += 1
