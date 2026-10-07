# Implementation decisions

Sources: [arXiv:2609.36145](https://arxiv.org/abs/2609.36145) (the EKI paper) and `EKI on 8 GB VRAM Implementation Roadmap.pdf`.
The roadmap's "Gaps in the paper" table is adopted as-is; everything below either adds to it
or overrides it. Every choice that can vary is a key in `configs/default.yaml`.

## Overrides of the roadmap

### Baseline / M_pre trainable parameters (2026-10-07)
- **Roadmap:** M_pre = Stage 1 settings (ViT LoRA + projector, LLM frozen), global data, 1 epoch.
- **Decision:** Baseline = M_pre = ViT **frozen**, LLM LoRA (r=32, alpha=64) + projector trained,
  global-localisation data only, loss = L_LM.
- **Why (from the paper):**
  - Sec. IV-B2 builds M_pre with "the standard MLLM baseline (fine-tuned for the global
    localization task)", so M_pre and the Baseline are the same model.
  - Sec. V-C1 defines the FGRA-only row of Table V as "vision encoder kept frozen at its
    pre-trained weights, LLM and projector fine-tuned with L_LM + L_FGRA". Table V's rows are
    toggles over one recipe, so the Baseline (no ticks) is that recipe without L_FGRA.
  - Fig. 7(b) shows that "supervised fine-tuning" changes LLM layer-1 features, which needs a
    trained LLM.
  - Side effect: Baseline vs FGRA-only now isolates L_FGRA (no LLM-LoRA confound).

### Table V rows as one recipe with three toggles
Every ablation row = optional Stage 1 (global + optional TF + optional IF; ViT LoRA + projector,
LLM frozen) followed by Stage 2 (ViT frozen, LLM LoRA + projector, global data, L_LM + optional L_FGRA).

| Row       | Stage 1 data       | Stage 2 loss      |
|-----------|--------------------|-------------------|
| Baseline  | skipped            | L_LM              |
| TF        | global + TF        | L_LM              |
| IF        | global + IF        | L_LM              |
| TF+IF     | global + TF + IF   | L_LM              |
| FGRA only | skipped            | L_LM + L_FGRA     |
| Full EKI  | global + TF + IF   | L_LM + L_FGRA     |

The paper does not say whether the TF/IF-only rows went through Stage 2. Running Stage 2 with
FGRA off keeps the trainable parameters identical across rows.

## Environment
- GPU: RTX 5060 Laptop (Blackwell, sm_120, 8 GB, bf16 supported → no fp16 fallback needed).
- Two conda envs: `eki` (Python 3.11, torch cu128, transformers/peft/bitsandbytes) and
  `eki-ocr` (PaddleOCR, used only for the one-off Phase 3 OCR pass).

## Code layout
- Overrides the roadmap's single notebook: an `eki/` package plus one CLI script per phase in
  `scripts/`, run in the background and resumable from step checkpoints.
- `notebooks/eki.ipynb` handles gate checks, spot-check overlays and figures.

## Data
- DocTamper is not available yet. `scripts/make_synthetic.py` writes a small LMDB in DocTamper's
  format (`image-%09d` JPEG, `label-%09d` PNG mask) so every phase can be smoke-tested.

## Choices made while implementing (2026-10-08)
- **Stage 1 mix** (`eki/samples.py:mix`): each source contributes min(its size, weight x |global|)
  samples, without repetition; if the total exceeds `stage1.max_samples`, all sources are scaled down.
  This is how "equal sampling weight per source" is implemented.
- **IF crops** are cut from the *original-resolution* image (then resized to 504), so a crop is a
  real magnification rather than an upsampled 504 crop. Crops are square, so there is no aspect distortion.
  The authentic crop is a random same-size window with no tampered pixels.
- **IF targets** are "Tampered." / "Real." with no boxes (paper Sec. IV-B3, Local Classification).
- **TF queries**: OCR lines (not words). A line is "tampered" if >= `min_box_area` mask pixels fall
  inside it, and "real" only if none do; partial overlaps below that threshold are dropped as
  ambiguous. 4 queries per image, target 40% tampered.
- **OCR**: PP-OCRv5 mobile det/rec on GPU (paddlepaddle-gpu 3.3.1 cu129, ~0.3 s/img). CPU costs ~9.5 s/img
  (mobile) or ~17 s/img (server models); oneDNN is disabled because of a paddle 3.3 bug.
- **Table V rows** all train on `ablation_ids.txt` (5k), so the rows share one training set; the
  headline Baseline/EKI runs use the 15k subset. M_pre for hard mining = the headline Baseline.
- **Answer length**: `train.max_len` 704 fits 324 image tokens + prompt + 16 boxes. Images with more
  components keep the 16 largest boxes (`data.max_boxes`).
- **Expert batch** 4 x 2 accumulation (same effective batch 8); Phase 0 measured 1.9 GB at batch 2.
- **Trainable precision**: LoRA, projector and phi weights are fp32 masters under bf16 autocast;
  the frozen ViT stays bf16 and the LLM stays NF4.
