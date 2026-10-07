"""Phase 1: box targets for every split, training subset + validation split, overlay check.

Writes  boxes.jsonl        {"id", "boxes", "tampered_frac"} for train and test splits
        subset_ids.txt     training subset (stratified by #boxes and tampered area)
        val_ids.txt        fixed validation split, disjoint from the subset
        ablation_ids.txt   smaller subset for ablation rows
        p1_overlays/       50 images with GT boxes drawn over the mask
Gate: mean IoU between rasterised boxes and the original mask, printed and saved.
"""
import json
import random
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from eki import boxes as B
from eki.config import cli
from eki.data import Splits, write_ids, write_jsonl


def stratum(boxes, frac):
    nb = min(len(boxes), 3)
    area_bin = 0 if frac == 0 else 1 if frac < 0.005 else 2 if frac < 0.02 else 3
    return nb, area_bin


def main():
    cfg, _ = cli(__doc__)
    S, size, work = Splits(cfg), cfg.data.img_size, cfg.work
    rows, ious, n_real = [], [], defaultdict(int)
    for split in [cfg.data.train_lmdb, *cfg.data.test_lmdbs]:
        db = S.db(split)
        for i in tqdm(range(len(db)), desc=split):
            _, mask = db.get(i, size)
            bx = B.mask_to_boxes(mask, cfg.data.min_box_area, cfg.data.merge_gap)
            if len(bx) > cfg.data.max_boxes:
                bx = B.mask_to_boxes(mask, cfg.data.min_box_area, cfg.data.merge_gap, cfg.data.max_boxes)
            rows.append({"id": f"{split}:{i}", "boxes": bx, "tampered_frac": float(mask.mean())})
            n_real[split] += not bx
            if mask.any():
                ious.append(B.pixel_iou_f1(B.rasterise(bx, size), mask)[0])
    write_jsonl(work / "boxes.jsonl", rows)

    train = [r for r in rows if r["id"].startswith(cfg.data.train_lmdb + ":")]
    rng = random.Random(cfg.seed)
    rng.shuffle(train)
    n_val = min(cfg.data.val_size, len(train) // 10)
    val, rest = train[:n_val], train[n_val:]
    strata = defaultdict(list)
    for r in rest:
        strata[stratum(r["boxes"], r["tampered_frac"])].append(r["id"])
    subset = _stratified(strata, min(cfg.data.train_subset, len(rest)), rng)
    abl = _stratified({k: [i for i in v if i in set(subset)] for k, v in strata.items()},
                      min(cfg.data.ablation_subset, len(subset)), rng)
    write_ids(work / "subset_ids.txt", subset)
    write_ids(work / "val_ids.txt", [r["id"] for r in val])
    write_ids(work / "ablation_ids.txt", abl)

    by_id = {r["id"]: r for r in rows}
    sub_real = sum(not by_id[i]["boxes"] for i in subset)
    _overlays(S, by_id, rng.sample(subset, min(50, len(subset))), work / "p1_overlays", size)

    report = {
        "box_vs_mask_iou_mean": float(np.mean(ious)) if ious else None,
        "box_vs_mask_iou_p10": float(np.percentile(ious, 10)) if ious else None,
        "real_images_per_split": dict(n_real),
        "subset": len(subset), "val": len(val), "ablation": len(abl),
        "subset_real_fraction": sub_real / max(1, len(subset)),
        "max_boxes_seen": max(len(r["boxes"]) for r in rows),
        "strata": {str(k): len(v) for k, v in sorted(strata.items())},
    }
    (work / "p1_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if report["subset_real_fraction"] < 0.05:
        print("NOTE: <5% authentic images in the subset; 'Real.' targets will come almost only "
              "from TF queries and IF crops (roadmap 'Still open': too few authentic examples).")


def _stratified(strata, n, rng):
    total = sum(len(v) for v in strata.values())
    out = []
    for v in strata.values():
        k = round(n * len(v) / max(1, total))
        out += rng.sample(v, min(k, len(v)))
    left = [i for v in strata.values() for i in v if i not in set(out)]
    out += rng.sample(left, max(0, min(n - len(out), len(left))))
    return sorted(out[:n], key=lambda s: int(s.rsplit(":", 1)[1]))


def _overlays(S, by_id, ids, out, size):
    out.mkdir(parents=True, exist_ok=True)
    for sid in ids:
        img, mask = S.get(sid)
        tint = np.array(img).copy()
        tint[mask] = (0.5 * tint[mask] + [127, 0, 0]).astype(np.uint8)
        im = Image.fromarray(tint)
        d = ImageDraw.Draw(im)
        for b in by_id[sid]["boxes"]:
            d.rectangle([b[0], b[1], b[2] - 1, b[3] - 1], outline=(0, 200, 0), width=1)
        im.save(out / (sid.replace(":", "_") + ".png"))


if __name__ == "__main__":
    main()
