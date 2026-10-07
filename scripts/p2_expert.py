"""Phase 2: train the forensic expert, score it on the validation split, cache FGRA targets.

  python scripts/p2_expert.py train   # -> expert/expert_full.pt (+ checkpoints), expert_backbone.pt
  python scripts/p2_expert.py eval    # -> expert/val_report.json (box-level pixel IoU/F1, Phase 8 metric)
  python scripts/p2_expert.py cache   # -> feats/{id}.npy, 324 x 1024 fp16 per image (subset + val)
"""
import json
import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from eki import boxes as B
from eki.config import cli
from eki.data import Splits, read_ids
from eki.expert import Expert, bce_dice, fgra_targets, to_tensor
from eki.train_utils import Checkpointer, JsonlLog, cosine_with_warmup


class MaskDS(Dataset):
    def __init__(self, cfg, ids):
        self.S, self.ids = Splits(cfg), ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img, mask = self.S.get(self.ids[i])
        return img, mask, self.ids[i]


def collate(b):
    imgs, masks, ids = zip(*b)
    return to_tensor(imgs), torch.from_numpy(np.stack(masks)).float().unsqueeze(1), list(ids)


def train(cfg):
    E, out = cfg.expert, cfg.work / "expert"
    ids = read_ids(cfg.work / "subset_ids.txt")
    model = Expert(cfg.model.expert, E.freeze_blocks, E.head_channels).cuda()
    model.backbone.gradient_checkpointing_enable()
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"expert trainable params: {sum(p.numel() for p in params) / 1e6:.1f}M")
    opt = torch.optim.AdamW(params, lr=E.lr, betas=tuple(cfg.train.betas))
    steps_per_epoch = math.ceil(len(ids) / (E.batch_size * E.grad_accum))
    total = steps_per_epoch * E.epochs
    sched = cosine_with_warmup(opt, total, int(cfg.train.warmup_ratio * total))
    ck = Checkpointer(out, cfg.train.save_every)
    state = ck.resume(model, opt, sched)
    log = JsonlLog(out / "log.jsonl")
    dl_kw = dict(batch_size=E.batch_size, num_workers=cfg.train.num_workers, collate_fn=collate)

    step = state["step"]
    for epoch in range(state["epoch"], E.epochs):
        g = torch.Generator().manual_seed(cfg.seed + epoch)
        order = torch.randperm(len(ids), generator=g).tolist()
        skip = state["micro"] if epoch == state["epoch"] else 0
        ds = MaskDS(cfg, [ids[i] for i in order[skip:]])
        model.train()
        t0, micro = time.time(), skip
        for x, y, _ in tqdm(DataLoader(ds, **dl_kw), desc=f"expert ep{epoch}"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x.cuda(non_blocking=True))
            loss = bce_dice(logits.float(), y.cuda()) / E.grad_accum
            loss.backward()
            micro += len(x)
            if micro % (E.batch_size * E.grad_accum) == 0 or micro == len(ids):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % cfg.train.log_every == 0:
                    log.write(step=step, epoch=epoch, loss=loss.item() * E.grad_accum,
                              lr=sched.get_last_lr()[0], s_per_img=(time.time() - t0) / max(1, micro - skip),
                              mem_gb=torch.cuda.max_memory_allocated() / 2**30)
                ck.maybe_save(step, model, opt, sched, epoch=epoch, micro=micro)
        ck.save(step, model, opt, sched, epoch=epoch + 1, micro=0)
    torch.save(model.state_dict(), out / "expert_full.pt")
    torch.save(model.backbone.state_dict(), cfg.work / "expert_backbone.pt")
    print("saved", out / "expert_full.pt")


def load_trained(cfg, backbone_only=False):
    model = Expert(cfg.model.expert, cfg.expert.freeze_blocks, cfg.expert.head_channels)
    if backbone_only:
        model.backbone.load_state_dict(torch.load(cfg.work / "expert_backbone.pt", map_location="cpu"))
    else:
        model.load_state_dict(torch.load(cfg.work / "expert" / "expert_full.pt", map_location="cpu"))
    return model.cuda().eval().to(torch.bfloat16)


@torch.no_grad()
def evaluate(cfg):
    model = load_trained(cfg)
    ids = read_ids(cfg.work / "val_ids.txt")
    size, ious, f1s = cfg.data.img_size, [], []
    dl = DataLoader(MaskDS(cfg, ids), batch_size=4, num_workers=cfg.train.num_workers, collate_fn=collate)
    for x, y, _ in tqdm(dl, desc="expert eval"):
        prob = torch.sigmoid(model(x.cuda().to(torch.bfloat16)).float()).cpu().numpy()[:, 0]
        for p, m in zip(prob, y.numpy()[:, 0] > 0):
            pb = B.mask_to_boxes(p > 0.5, cfg.data.min_box_area)
            gb = B.mask_to_boxes(m, cfg.data.min_box_area)
            iou, f1 = B.pixel_iou_f1(B.rasterise(pb, size), B.rasterise(gb, size))
            ious.append(iou); f1s.append(f1)
    rep = {"val_box_iou": float(np.mean(ious)), "val_box_f1": float(np.mean(f1s)), "n": len(ious)}
    (cfg.work / "expert" / "val_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


@torch.no_grad()
def cache(cfg):
    model = load_trained(cfg, backbone_only=True)
    out = cfg.work / "feats"
    out.mkdir(exist_ok=True)
    ids = read_ids(cfg.work / "subset_ids.txt") + read_ids(cfg.work / "val_ids.txt")
    ids = [i for i in ids if not (out / f"{i.replace(':', '_')}.npy").exists()]
    dl = DataLoader(MaskDS(cfg, ids), batch_size=4, num_workers=cfg.train.num_workers, collate_fn=collate)
    for x, _, bid in tqdm(dl, desc="cache feats"):
        f = fgra_targets(model, x.cuda().to(torch.bfloat16)).half().cpu().numpy()
        for sid, a in zip(bid, f):
            np.save(out / f"{sid.replace(':', '_')}.npy", a)
    print(f"cached {len(ids)} feature files in {out}")


def feat_path(cfg, sid):
    return cfg.work / "feats" / f"{sid.replace(':', '_')}.npy"


if __name__ == "__main__":
    cfg, args = cli(__doc__, lambda p: p.add_argument("cmd", choices=["train", "eval", "cache"]))
    {"train": train, "eval": evaluate, "cache": cache}[args.cmd](cfg)
