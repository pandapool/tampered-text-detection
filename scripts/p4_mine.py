"""Phase 4b: hard-sample mining with M_pre and Image-Focused crops (paper IV-B2).

Needs predictions of M_pre on the training subset first:
  python scripts/infer.py --run baseline --ids subset_ids.txt
  (cross-fit: train baseline_a on half_a.txt, baseline_b on half_b.txt, run each on the other half;
   `python scripts/p4_mine.py split` writes the halves)
Then:
  python scripts/p4_mine.py mine --preds runs/baseline/preds_subset_ids.jsonl [more preds files]

Hard = pixel IoU between the union of predicted and union of GT boxes < mining.iou_thresh
(no prediction = 0). For each hard image, every GT box is cropped from the *original-resolution*
image with context padding (>= crop_scale x box, >= crop_min px in the img_size frame), plus an
equal number of random authentic crops of matching size; all resized to img_size.
Writes hard_ids.txt, patches/*.png, patches/index.jsonl {"id", "path", "tampered"}.
"""
import json
import random

from PIL import Image

from eki import boxes as B
from eki.config import cli
from eki.data import Splits, read_ids, read_jsonl, write_ids, write_jsonl


def split_halves(cfg):
    ids = read_ids(cfg.work / "subset_ids.txt")
    rng = random.Random(cfg.seed)
    rng.shuffle(ids)
    h = len(ids) // 2
    write_ids(cfg.work / "half_a.txt", sorted(ids[:h]))
    write_ids(cfg.work / "half_b.txt", sorted(ids[h:]))
    print(f"half_a: {h}, half_b: {len(ids) - h}")


def crop_box(b, size, M):
    """Square crop around box b, side >= crop_scale * max(w, h) and >= crop_min, inside the frame."""
    w, h = b[2] - b[0], b[3] - b[1]
    side = int(min(size, max(M.crop_scale * max(w, h), M.crop_min)))
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    x1 = int(min(max(cx - side / 2, 0), size - side))
    y1 = int(min(max(cy - side / 2, 0), size - side))
    return [x1, y1, x1 + side, y1 + side]


def random_authentic(mask, side, rng, size, tries=50):
    for _ in range(tries):
        x1, y1 = rng.randint(0, size - side), rng.randint(0, size - side)
        if not mask[y1:y1 + side, x1:x1 + side].any():
            return [x1, y1, x1 + side, y1 + side]
    return None


def cut(raw, c, size):
    """Crop c (img_size frame) from the original-resolution image, then resize to img_size."""
    W, H = raw.size
    box = (c[0] * W / size, c[1] * H / size, c[2] * W / size, c[3] * H / size)
    return raw.crop(tuple(round(v) for v in box)).resize((size, size), Image.BICUBIC)


def mine(cfg, pred_files):
    S, M, size = Splits(cfg), cfg.mining, cfg.data.img_size
    rng = random.Random(cfg.seed)
    gt = {r["id"]: r["boxes"] for r in read_jsonl(cfg.work / "boxes.jsonl")}
    preds = {r["id"]: r["boxes"] for f in pred_files for r in read_jsonl(cfg.work / f)}
    ious = {sid: B.box_iou_union(pb, gt[sid], size) for sid, pb in preds.items() if gt[sid]}
    hard = sorted(sid for sid, v in ious.items() if v < M.iou_thresh)
    write_ids(cfg.work / "hard_ids.txt", hard)

    out = cfg.work / "patches"; out.mkdir(exist_ok=True)
    index = []
    for sid in hard:
        split, idx = sid.rsplit(":", 1)
        raw, _ = S.db(split).raw(int(idx))
        _, mask = S.get(sid)
        stem = sid.replace(":", "_")
        for k, b in enumerate(gt[sid]):
            c = crop_box(b, size, M)
            cut(raw, c, size).save(out / f"{stem}_t{k}.png")
            index.append({"id": sid, "path": f"patches/{stem}_t{k}.png", "tampered": True, "crop": c})
            a = random_authentic(mask, c[2] - c[0], rng, size)
            if a:
                cut(raw, a, size).save(out / f"{stem}_r{k}.png")
                index.append({"id": sid, "path": f"patches/{stem}_r{k}.png", "tampered": False, "crop": a})
    write_jsonl(out / "index.jsonl", index)
    rep = {"evaluated": len(ious), "hard": len(hard), "hard_fraction": len(hard) / max(1, len(ious)),
           "patches_tampered": sum(r["tampered"] for r in index),
           "patches_real": sum(not r["tampered"] for r in index)}
    (cfg.work / "p4_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    if rep["hard_fraction"] < 0.05 and not M.cross_fit:
        print("NOTE: few hard samples; M_pre has seen these images. Consider mining.cross_fit "
              "(python scripts/p4_mine.py split, then train/infer per half).")


if __name__ == "__main__":
    def args_fn(p):
        p.add_argument("cmd", choices=["split", "mine"])
        p.add_argument("--preds", nargs="*", default=["runs/baseline/preds_subset_ids.jsonl"])
    cfg, a = cli(__doc__, args_fn)
    split_halves(cfg) if a.cmd == "split" else mine(cfg, a.preds)
