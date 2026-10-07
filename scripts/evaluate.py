"""Phase 8: pixel IoU/F1 per image from rasterised boxes, averaged per dataset (paper V-A5).

  python scripts/evaluate.py --runs baseline tf if tfif fgra_only eki

Reads runs/<run>/preds_<split>.jsonl for every split in data.test_lmdbs, writes results.csv with
one row per (run, split): IoU/F1 over all images and over tampered images only. Checks the gate
ordering Baseline < TF+IF < FGRA only < full EKI (ablation runs) on the first test split.
"""
import numpy as np
import pandas as pd

from eki import boxes as B
from eki.config import cli
from eki.data import read_jsonl

GATE = ["baseline_a", "tfif", "fgra_only", "eki_a"]  # Table V rows, same training ids


def score(preds, gt_boxes, size):
    rows = []
    for r in preds:
        gb = gt_boxes[r["id"]]
        iou, f1 = B.pixel_iou_f1(B.rasterise(r["boxes"], size), B.rasterise(gb, size))
        rows.append({"id": r["id"], "iou": iou, "f1": f1, "tampered": bool(gb), "ok": r["ok"]})
    return pd.DataFrame(rows)


def bootstrap_ci(x, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    cfg, a = cli(__doc__, lambda p: p.add_argument("--runs", nargs="+", required=True))
    size = cfg.data.img_size
    gt = {r["id"]: r["boxes"] for r in read_jsonl(cfg.work / "boxes.jsonl")}
    table = []
    for run in a.runs:
        for split in cfg.data.test_lmdbs:
            p = cfg.work / "runs" / run / f"preds_{split}.jsonl"
            if not p.exists():
                continue
            df = score(read_jsonl(p), gt, size)
            t = df[df.tampered]
            df.to_csv(p.with_suffix(".scores.csv"), index=False)
            lo, hi = bootstrap_ci(df.iou.values)
            table.append({"run": run, "split": split, "n": len(df), "iou": df.iou.mean(), "f1": df.f1.mean(),
                          "iou_ci95": f"[{lo:.3f}, {hi:.3f}]",
                          "iou_tampered_only": t.iou.mean() if len(t) else np.nan,
                          "f1_tampered_only": t.f1.mean() if len(t) else np.nan,
                          "parse_fail": 1 - df.ok.mean()})
    res = pd.DataFrame(table)
    res.to_csv(cfg.work / "results.csv", index=False)
    print(res.to_string(index=False, float_format="%.3f"))

    main_split = cfg.data.test_lmdbs[0]
    have = res[(res.split == main_split) & res.run.isin(GATE)].set_index("run")
    seq = [r for r in GATE if r in have.index]
    if len(seq) > 1:
        vals = [have.loc[r, "iou"] for r in seq]
        ok = all(x < y for x, y in zip(vals, vals[1:]))
        print(f"\nGate ({main_split} IoU): " + " < ".join(f"{r}={v:.3f}" for r, v in zip(seq, vals))
              + ("  PASS" if ok else "  FAIL"))


if __name__ == "__main__":
    main()
