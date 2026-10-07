"""Phase 3b: Text-Focused instruction pairs from OCR boxes (paper IV-B1).

For each OCR line with conf > ocr.conf_thresh: if it overlaps the tampered mask, the target is
"Tampered. b_gt" with b_gt the box(es) of the mask *inside* the OCR region, else "Real.".
Samples ocr.queries_per_image queries per image, balancing real/tampered. Writes tf_pairs.jsonl
{"id", "text", "box", "gt"} and p3_spotcheck/ (30 drawn pairs).
Gate: tampered fraction in roughly 30-50%.
"""
import json
import random

import numpy as np
from PIL import ImageDraw

from eki import boxes as B
from eki.config import cli
from eki.data import Splits, read_jsonl, write_jsonl


def main():
    cfg, _ = cli(__doc__)
    S, O, rng = Splits(cfg), cfg.ocr, random.Random(cfg.seed)
    min_px = cfg.data.min_box_area
    pairs = []
    for r in read_jsonl(cfg.work / "ocr.jsonl"):
        lines = [l for l in r["lines"] if l["conf"] > O.conf_thresh and l["text"].strip()]
        if not lines:
            continue
        _, mask = S.get(r["id"])
        tam, real = [], []
        for l in lines:
            x1, y1, x2, y2 = l["box"]
            sub = mask[y1:y2, x1:x2]
            if sub.sum() >= min_px:
                gt = [[b[0] + x1, b[1] + y1, b[2] + x1, b[3] + y1]
                      for b in B.mask_to_boxes(sub, min_px, cfg.data.merge_gap, cfg.data.max_boxes)]
                tam.append({"id": r["id"], "text": l["text"], "box": l["box"], "gt": gt})
            elif sub.sum() == 0:  # partial overlaps below min_px are ambiguous; skip them
                real.append({"id": r["id"], "text": l["text"], "box": l["box"], "gt": []})
        k = O.queries_per_image
        n_t = min(len(tam), max(1, round(k * O.tampered_frac_target)) if tam else 0)
        pairs += rng.sample(tam, n_t) + rng.sample(real, min(len(real), k - n_t))
    write_jsonl(cfg.work / "tf_pairs.jsonl", pairs)

    frac = sum(bool(p["gt"]) for p in pairs) / max(1, len(pairs))
    rep = {"pairs": len(pairs), "tampered_fraction": frac,
           "images": len({p["id"] for p in pairs})}
    (cfg.work / "p3_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    if not 0.3 <= frac <= 0.5:
        print("GATE WARN: tampered fraction outside 30-50%; adjust ocr.tampered_frac_target")

    out = cfg.work / "p3_spotcheck"; out.mkdir(exist_ok=True)
    for i, p in enumerate(rng.sample(pairs, min(30, len(pairs)))):
        img, _ = S.get(p["id"])
        d = ImageDraw.Draw(img)
        d.rectangle(p["box"], outline=(0, 0, 255), width=2)
        for b in p["gt"]:
            d.rectangle([b[0], b[1], b[2] - 1, b[3] - 1], outline=(255, 0, 0), width=1)
        img.save(out / f"{i:02d}_{'T' if p['gt'] else 'R'}.png")


if __name__ == "__main__":
    main()
