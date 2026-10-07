# EKI on 8 GB VRAM

Scaled-down reproduction of **Expert Knowledge Internalization**
([arXiv:2609.36145](https://arxiv.org/abs/2609.36145)) following
`EKI on 8 GB VRAM Implementation Roadmap.pdf`: Qwen2.5-VL-3B with an NF4 LLM and a bf16 ViT,
a DINOv2-L forensic expert, Stage 1 (Precise Spatial Focus), Stage 2 (FGRA), box inference
and pixel IoU/F1 evaluation. Decisions that go beyond or override the roadmap are recorded in
[`DECISIONS.md`](DECISIONS.md).

## Setup

```bash
conda create -n eki python=3.11 && conda activate eki
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # Blackwell needs cu128+
pip install "transformers>=5" peft "bitsandbytes>=0.45" accelerate qwen-vl-utils \
    opencv-python-headless scikit-image lmdb pyyaml tqdm pandas matplotlib ipykernel nbformat pytest
pip install -e .

conda create -n eki-ocr python=3.11            # PaddleOCR lives apart from torch's CUDA libs
conda run -n eki-ocr pip install paddleocr lmdb pyyaml pillow numpy tqdm && conda run -n eki-ocr pip uninstall -y paddlepaddle
conda run -n eki-ocr pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
conda run -n eki-ocr pip install -e . --no-deps
```

## Data

Put the DocTamper LMDBs under `data/doctamper/{TrainingSet,TestingSet,FCD,SCD}` (or set
`data.root`). Until then, `python scripts/make_synthetic.py` writes a small look-alike to
`data/synthetic/` that `configs/smoke.yaml` uses.

## Running

Everything is driven by `scripts/run_all.sh`. Each step skips work that is already done, and
training resumes mid-epoch from `runs/<name>/ckpt.pt`, so you can re-run it after a crash.

```bash
CONFIG=configs/smoke.yaml   scripts/run_all.sh     # synthetic, end to end in ~20 min
CONFIG=configs/default.yaml scripts/run_all.sh     # the real thing
STEPS="ablation eval sweep" scripts/run_all.sh     # selected steps only
nohup scripts/run_all.sh > run.log 2>&1 &          # long runs
```

| Step       | Script                                  | Roadmap phase | Main outputs (under `work_dir`)          |
|------------|-----------------------------------------|---------------|------------------------------------------|
| `probe`    | `p0_probe.py`                           | 0             | `p0_report.json`                         |
| `data`     | `p1_data.py`                            | 1             | `boxes.jsonl`, `subset_ids.txt`, `val_ids.txt`, `ablation_ids.txt` |
| `expert`   | `p2_expert.py train/eval/cache`         | 2             | `expert_backbone.pt`, `feats/*.npy`      |
| `ocr`      | `p3_ocr.py` (eki-ocr env), `p3_tf.py`   | 3             | `ocr.jsonl`, `tf_pairs.jsonl`            |
| `baseline` | `train_mllm.py --stage 2` (no FGRA)     | 4             | `runs/baseline/`                         |
| `mine`     | `infer.py`, `p4_mine.py`                | 4             | `hard_ids.txt`, `patches/`               |
| `stage1`   | `train_mllm.py --stage 1`               | 5             | `runs/stage1/visual.pt`                  |
| `eki`      | `train_mllm.py --stage 2 --vit stage1`  | 6             | `runs/eki/`                              |
| `infer`    | `infer.py`                              | 7             | `runs/*/preds_<split>.jsonl`             |
| `ablation` | Table V rows on `ablation_ids.txt`      | 8             | `runs/{baseline_a,tf,if,tfif,fgra_only,eki_a}/` |
| `sweep`    | FGRA layer in {0, 1, 4, 14, last}       | 8 (Fig. 6)    | `runs/fgra_l*/`                          |
| `eval`     | `evaluate.py`, `figures.py`             | 8             | `results.csv`, `fig_*.png`               |

Every value lives in `configs/default.yaml`. Override from the CLI with
`--set key=value ...` or via `EXTRA="key=value"` for `run_all.sh`.
`notebooks/eki.ipynb` reads the reports and logs for each gate; it never trains.

## Measured on an RTX 5060 Laptop (8 GB)

| Config                                  | Peak allocated | Time per sample |
|-----------------------------------------|----------------|-----------------|
| Stage 1 (ViT LoRA + projector, NF4 LLM) | 4.2 GB (4.7 with optimiser) | 0.80 s |
| Stage 2 (LLM LoRA + projector + phi)    | 4.7 GB (5.3 with optimiser) | 0.83 s |
| Expert (DINOv2-L, last 12 blocks, b=2)  | 1.9 GB         | 0.11 s          |
| Inference (greedy, NF4)                 | -              | 1.0-1.3 s/image |
| OCR, PP-OCRv5 mobile, GPU (CPU)         | -              | ~0.3 s/image (CPU ~9.5 s) |
