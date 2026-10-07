"""Phase 8 figures.

  python scripts/figures.py pca --runs original baseline eki --ids TestingSet:0 TestingSet:5
      Fig. 7(b)-style: LLM layer-l visual-token hidden states -> top-3 principal components ->
      min-max normalised RGB over the 18x18 token grid, next to the image and GT mask.
  python scripts/figures.py sweep
      Fig. 6-style: IoU vs FGRA layer from results.csv rows named fgra_l<k> (+ eki_a as l=1).
"""
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from eki import mllm
from eki.config import cli
from eki.data import Splits, read_jsonl
from eki.samples import messages


@torch.no_grad()
def pca_grid(model, proc, img, prompt, layer, size):
    text = proc.apply_chat_template(messages(prompt), tokenize=False, add_generation_prompt=True)
    enc = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
    o = model(**enc, output_hidden_states=True)
    h = mllm.visual_hidden(o.hidden_states[layer], enc["input_ids"], mllm.image_token_id(model))[0].float()
    h = h - h.mean(0)
    _, _, v = torch.pca_lowrank(h, q=3, center=False)
    p = h @ v[:, :3]
    p = (p - p.min(0).values) / (p.max(0).values - p.min(0).values + 1e-6)
    g = int(round(p.shape[0] ** 0.5))
    arr = (p.view(g, g, 3).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr).resize((size, size), Image.NEAREST)


def pca(cfg, runs, ids, layer):
    S, size = Splits(cfg), cfg.data.img_size
    proc = mllm.processor(cfg)
    gt = {r["id"]: r["boxes"] for r in read_jsonl(cfg.work / "boxes.jsonl")}
    cols = {sid: [] for sid in ids}
    for sid in ids:
        img, mask = S.get(sid)
        im = img.copy(); d = ImageDraw.Draw(im)
        for b in gt.get(sid, []):
            d.rectangle([b[0], b[1], b[2] - 1, b[3] - 1], outline=(255, 0, 0), width=2)
        cols[sid] += [im, Image.fromarray(mask.astype(np.uint8) * 255).convert("RGB")]
    for run in runs:
        if run == "original":
            model = mllm.load_base(cfg).eval()
        else:
            model = mllm.load_for_inference(cfg, cfg.work / "runs" / run)
        for sid in ids:
            cols[sid].append(pca_grid(model, proc, S.image(sid), cfg.prompts["global"], layer, size))
        del model; torch.cuda.empty_cache()
    rows = [np.concatenate([np.asarray(c) for c in v], 1) for v in cols.values()]
    out = cfg.work / f"fig_pca_layer{layer}.png"
    Image.fromarray(np.concatenate(rows, 0)).save(out)
    print("columns: image, mask,", ", ".join(runs), "->", out)


def sweep(cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = pd.read_csv(cfg.work / "results.csv")
    split = cfg.data.test_lmdbs[0]
    rows = res[(res.split == split) & res.run.str.match(r"^(fgra_l\d+|eki_a)$")].copy()
    rows["layer"] = rows.run.map(lambda r: 1 if r == "eki_a" else int(r.split("_l")[1]))
    rows = rows.sort_values("layer")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(rows.layer.astype(str), rows.iou, marker="o", label="IoU")
    ax.plot(rows.layer.astype(str), rows.f1, marker="s", label="F1")
    ax.set_xlabel("LLM layer aligned by FGRA"); ax.set_title(split); ax.legend()
    fig.tight_layout(); fig.savefig(cfg.work / "fig_layer_sweep.png", dpi=150)
    print(rows[["run", "layer", "iou", "f1"]].to_string(index=False))


if __name__ == "__main__":
    def args_fn(p):
        p.add_argument("cmd", choices=["pca", "sweep"])
        p.add_argument("--runs", nargs="*", default=["original", "baseline", "eki"])
        p.add_argument("--ids", nargs="*", default=None)
        p.add_argument("--layer", type=int, default=1)
    cfg, a = cli(__doc__, args_fn)
    if a.cmd == "pca":
        ids = a.ids or [f"{cfg.data.test_lmdbs[0]}:{i}" for i in range(4)]
        pca(cfg, a.runs, ids, a.layer)
    else:
        sweep(cfg)
