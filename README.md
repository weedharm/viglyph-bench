# ViGlyph-Bench

Benchmark, evaluation code and results for the paper

> **Do OCR Rewards Teach Diacritics? Region-Anchored Reinforcement Learning and Faithful Evaluation for Vietnamese Text Rendering**
> Đồng Quốc Hoàn (quochoan.hust.k63@gmail.com)

ViGlyph-Bench measures how well text-to-image models render Vietnamese text, counting errors at the level of single diacritics. The repository contains the prompts, the diacritic-aware metrics, the code for the three OCR readers and the RA-NFT training, all OCR outputs behind the numbers in the paper, and the annotations of the blind visual audit.

## Contents

```
bench/
  viglyph/prompts_v1.jsonl        500 test prompts (Vietnamese captions)
  viglyph/prompts_dev_v1.jsonl    60 dev prompts (checkpoint selection)
  viglyph/prompts_v1_en*.jsonl    the same 500 targets with English captions in the Z-LAN prompt format
  viglyph/v1-texts.txt            the 500 target texts
  train/prompts_rl_v1.jsonl       2,000 RL training prompts (no text shared with test/dev)
  english/                        90 English prompts (retention test)
  ocr_injection/                  error-injection test: injected strings and OCR outputs
code/
  src/vitext/                     metrics, OCR wrappers, RA-NFT objective, rollout, training utilities
  scripts/                        generation, OCR, scoring, paired tests, training, audit scoring
results/
  runs/<run>/                     OCR outputs (ocr_qwen3vl/vintern/easyocr.jsonl), generation logs, summaries
  paired/                         paired comparisons between runs under the three readers
  dev_three_readers_by_ckpt.json  dev-split scores per RL checkpoint (reward over-optimisation figure)
  visual_audit/                   blind visual audit: two passes (A, B), adjudication (C), pair key, summary
```

### Prompt format

Each line of `prompts_v1.jsonl` is a JSON object, for example:

```json
{"id": "vg0_0000", "text": "Phở Bò", "bucket": "short", "n_words": 2, "is_upper": false, "template": "sign_0",
 "prompt": "Ảnh chụp tấm biển hiệu cửa hàng ... dòng chữ \"Phở Bò\" ...", "seed": 20261005, "marks": {...}}
```

- `vg0_0000`–`vg0_0179` form the **test split** used during development.
- `vg0_0180`–`vg0_0499` are the **held-out extension**, written later and not used for any decision.
- `bucket` is short (1–3 words), medium (4–10) or long (11–25). `seed` is the fixed generation seed.

### Runs in `results/runs/`

| Run | Paper row |
|---|---|
| `v1_base`, `v1_rl80` | Qwen-Image-2.1 base and RA-NFT (iteration 80), all 500 prompts |
| `v1_qwenimage1` | Qwen-Image (2025), true CFG 4.0, 50 steps |
| `v1_zimage_base`, `v1_zimage_zlan` | Z-Image and Z-Image + Z-LAN Vietnamese LoRA (English caption format) |
| `baseline_1024` | Qwen-Image-2.1 base, test split |
| `eval_lora_l0_ck1000`, `eval_lora_l2bg_ck1000` | SFT and SFT + anchor |
| `eval_rl_nft_v3_ckpt-30`, `eval_rl_nft_v3b_ckpt-80`, `eval_rl_nft_v3c_ckpt-90` | RA-NFT at iterations 30, 80 and 90 |
| `eval_rl_faithful_ckpt-50` | RA-NFT-F (reward from Qwen3-VL and EasyOCR) |
| `eval_rl_abl_noregion_ck80`, `eval_rl_abl_noanchor_ck80`, `eval_rl_abl_nosym_ck30/80` | ablations |
| `bestofn_base`, `bestofn_rl_v3_ck80` | 4 seeds per test prompt (720 image pairs) |

OCR readers: Qwen3-VL-8B-Instruct (the reward reader), Vintern-1B-v3.5 (held out) and EasyOCR 1.7.2 with the Vietnamese model (held out, no language model).

## Reproducing the scores

Requirements: Python ≥ 3.10. Scoring needs only the standard library. Generation, OCR and training need PyTorch, diffusers (Qwen-Image-2.1 support), transformers, peft and easyocr.

```bash
cd code
# paired comparison of two runs under the three readers (exact match, CER_NFD, sign tests, error split)
python scripts/compare-two-runs-paired-across-ocrs.py --prompts ../bench/viglyph/prompts_v1.jsonl \
    --a ../results/runs/v1_base --b ../results/runs/v1_rl80
# visual audit summary (accuracy per model, McNemar test, agreement, OCR vs. audit)
python scripts/score-blind-visual-audit.py --key ../results/visual_audit/pair_key.json \
    --audit-dir ../results/visual_audit --prompts ../bench/viglyph/prompts_v1.jsonl
```

Main entry points for regeneration:
- `generate-images-qwen-image-21.py`, `generate-images-z-image-with-optional-lora.py` to generate images;
- `run-ocr-on-images.py` (Qwen3-VL and Vintern) and `ocr-images-with-easyocr.py` to read them;
- `score-baseline-from-ocr.py` to score the readings;
- `train-rl-nft-qwen-image-21.py` to train RA-NFT (flags `--symmetric-p --text-region-loss --w-base 0.5`).

Comments in the code are partly in Vietnamese.

## Visual audit

The blind visual audit was performed by a multimodal LLM (Claude Opus, Anthropic), not by human annotators.
- Each annotation pass saw the base and RA-NFT images of a prompt in random order as image 1 and image 2. `pair_key.json` maps image 1/2 back to the model.
- For each image, the annotator transcribed the drawn text from zoomed crops (`drawn`) and gave a label: `ok`, `diac` (diacritic error) or `char` (character error).
- Passes A and B are independent. Pass C adjudicates the 9 images on which A and B disagreed.

## Not included

- Generated images are not redistributed. Rerun the generation scripts with the fixed seeds to reproduce them.
- Model and LoRA weights are not redistributed: Qwen-Image-2.1, Qwen-Image, Z-Image, Z-LAN, Qwen3-VL and Vintern are available from their original sources under their own licenses.

## License

Code: MIT License (`LICENSE`). Benchmark prompts, OCR outputs and annotations: CC BY 4.0 (`LICENSE-DATA`).

## Citation

```bibtex
@article{dong2026ocrrewards,
  title  = {Do OCR Rewards Teach Diacritics? Region-Anchored Reinforcement Learning and Faithful Evaluation for Vietnamese Text Rendering},
  author = {Đồng Quốc Hoàn},
  year   = {2026},
  note   = {Under review}
}
```
