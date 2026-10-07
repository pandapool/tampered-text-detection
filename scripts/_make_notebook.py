"""Regenerates notebooks/eki.ipynb (kept as a script so the notebook diff stays reviewable)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb["cells"] = [
md("""# EKI on 8 GB: gates and figures
Heavy work runs in `scripts/` (see `scripts/run_all.sh`). This notebook only **reads** what those
scripts write under `work_dir`, so it can be re-run at any time without touching the GPU.
Set `CONFIG` to the config the scripts were run with."""),
code("""import json, glob, random
from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt
from PIL import Image
from eki.config import load

CONFIG = '../configs/smoke.yaml'   # or ../configs/default.yaml
cfg = load(CONFIG); W = cfg.work
def report(name):
    p = W / name
    return json.loads(p.read_text()) if p.exists() else f'{name}: not run yet'
def show(paths, n=6, cols=3, size=4):
    paths = sorted(paths)[:n]
    if not paths: return print('no images')
    rows = (len(paths) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols * size, rows * size), squeeze=False)
    for ax in axs.flat: ax.axis('off')
    for ax, p in zip(axs.flat, paths): ax.imshow(Image.open(p)); ax.set_title(Path(p).stem, fontsize=8)
    plt.tight_layout()
W"""),
md("## Phase 0: memory probe\nGate: every training peak < 7.2 GB; 324 visual tokens in row-major order."),
code("report('p0_report.json')"),
md("## Phase 1: box targets\nGate: rasterised boxes match the masks (IoU near 1); inspect overlays (green = box, red tint = mask). Check `subset_real_fraction`."),
code("display(report('p1_report.json'))\nshow(glob.glob(str(W / 'p1_overlays/*.png')), n=6)"),
md("## Phase 2: forensic expert\nGate (after Phase 4): expert val box IoU clearly above the Baseline."),
code("""display(report('expert/val_report.json'))
p = W / 'expert/log.jsonl'
if p.exists():
    df = pd.read_json(p, lines=True); df.plot(x='step', y='loss', title='expert loss')"""),
md("## Phase 3: OCR + Text-Focused pairs\nGate: tampered fraction 30-50%; spot-check (blue = OCR query box, red = GT box inside it)."),
code("display(report('p3_report.json'))\nshow(glob.glob(str(W / 'p3_spotcheck/*.png')), n=9)"),
md("## Phase 4: M_pre / Baseline and hard mining\nGate: hard fraction reported, crops look right."),
code("display(report('p4_report.json'))\nshow(glob.glob(str(W / 'patches/*.png')), n=8, cols=4, size=3)"),
md("## Phases 5-6: training curves\nStage 2: `cos` (FGRA cosine) should rise steadily while `l_lm` keeps falling."),
code("""runs = sorted(p.parent.name for p in (W / 'runs').glob('*/log.jsonl'))
fig, axs = plt.subplots(1, 3, figsize=(16, 4))
for r in runs:
    df = pd.read_json(W / 'runs' / r / 'log.jsonl', lines=True)
    axs[0].plot(df.step, df.l_lm, label=r)
    if 'cos' in df: axs[1].plot(df.step, df.cos, label=r)
    axs[2].plot(df.step, df.peak_gb, label=r)
for ax, t in zip(axs, ['L_LM', 'FGRA cosine', 'peak GB']): ax.set_title(t); ax.legend(fontsize=7)
runs"""),
md("## Phase 7: inference\nGate: parse failure < 1%. `ms_per_image` covers the latency claim (EKI vs Baseline, Table IV)."),
code("""rows = []
for p in sorted((W / 'runs').glob('*/preds_*.summary.json')):
    rows.append({'run': p.parent.name, 'split': p.name[len('preds_'):-len('.summary.json')], **json.loads(p.read_text())})
pd.DataFrame(rows)"""),
md("## Phase 8: results\nGate: Baseline < TF+IF < FGRA only < full EKI on the first test split (ablation rows)."),
code("""p = W / 'results.csv'
if p.exists():
    res = pd.read_csv(p)
    display(res.pivot_table(index='run', columns='split', values=['iou', 'f1']).round(3))
show(glob.glob(str(W / 'fig_*.png')), n=2, cols=1, size=10)"""),
md("### Qualitative: predictions vs GT (green = GT, red = prediction)"),
code("""from eki.data import Splits, read_jsonl
from PIL import ImageDraw
def draw(run, split, k=4):
    S = Splits(cfg); gt = {r['id']: r['boxes'] for r in read_jsonl(W / 'boxes.jsonl')}
    preds = read_jsonl(W / 'runs' / run / f'preds_{split}.jsonl')
    fig, axs = plt.subplots(1, k, figsize=(4 * k, 4))
    for ax, r in zip(axs, random.Random(0).sample(preds, k)):
        im = S.image(r['id']); d = ImageDraw.Draw(im)
        for b in gt[r['id']]: d.rectangle(b, outline=(0, 200, 0), width=2)
        for b in r['boxes']: d.rectangle(b, outline=(255, 0, 0), width=2)
        ax.imshow(im); ax.axis('off'); ax.set_title(r['text'][:40], fontsize=7)
# draw('eki', cfg.data.test_lmdbs[0])"""),
]
nb.metadata["kernelspec"] = {"name": "eki", "display_name": "Python (eki)", "language": "python"}
nbf.write(nb, "notebooks/eki.ipynb")
print("wrote notebooks/eki.ipynb")
