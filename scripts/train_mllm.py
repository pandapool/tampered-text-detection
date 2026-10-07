"""Phases 4-6: train the MLLM. One script, two stages, every Table V row is a config toggle.

  # Baseline / M_pre (Phase 4): Stage 2 recipe, original ViT, no FGRA, 1 epoch
  python scripts/train_mllm.py --stage 2 --name baseline --set stage2.use_fgra=false stage2.epochs=1
  # Stage 1 (Phase 5): global + TF + IF, ViT LoRA + projector
  python scripts/train_mllm.py --stage 1 --name stage1
  # Stage 2 (Phase 6): full EKI from the Stage 1 ViT
  python scripts/train_mllm.py --stage 2 --name eki --vit stage1

Outputs go to <work_dir>/runs/<name>/: ckpt.pt (resumable), log.jsonl, and final weights
(visual.pt for Stage 1; adapter/ + merger.pt + phi.pt + vit_from.txt for Stage 2).
Re-running the same command resumes from ckpt.pt.
"""
import json
import math
import time

import torch
from torch.utils.data import DataLoader

from eki import mllm
from eki.config import cli
from eki.data import read_ids, read_jsonl
from eki.samples import Collator, SFTDataset, global_samples, if_samples, mix, tf_samples
from eki.train_utils import Checkpointer, JsonlLog, cosine_with_warmup


def args_fn(p):
    p.add_argument("--stage", type=int, choices=[1, 2], required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--ids", default="subset_ids.txt", help="file under work_dir with training sample ids")
    p.add_argument("--vit", default=None, help="Stage 2: run name whose Stage 1 ViT to start from")
    p.add_argument("--max_steps", type=int, default=None, help="stop early (smoke tests)")


def build_samples(cfg, stage, ids):
    boxes = {r["id"]: r["boxes"] for r in read_jsonl(cfg.work / "boxes.jsonl")}
    glob = global_samples(cfg, ids, boxes)
    if stage == 2:
        return glob  # Stage 2 trains on global localisation only (DECISIONS / roadmap gaps table)
    idset = set(ids)
    src = {"global": glob}
    if cfg.stage1.use_tf:
        src["tf"] = [s for s in tf_samples(cfg) if s["id"] in idset]
    if cfg.stage1.use_if:
        src["if_"] = [s for s in if_samples(cfg) if s["id"] in idset]
    for k, v in src.items():
        print(f"stage1 source {k}: {len(v)} samples")
    return mix(src, dict(cfg.stage1.mix), cfg.stage1.max_samples, cfg.seed)


def main():
    cfg, a = cli(__doc__, args_fn)
    T = cfg.train
    out = cfg.work / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    ids = read_ids(cfg.work / a.ids)
    samples = build_samples(cfg, a.stage, ids)
    use_fgra = a.stage == 2 and cfg.stage2.use_fgra
    epochs = cfg.stage1.epochs if a.stage == 1 else cfg.stage2.epochs

    proc = mllm.processor(cfg)
    if a.stage == 1:
        model, phi = mllm.build_stage1(cfg), None
    else:
        vit_dir = cfg.work / "runs" / a.vit if a.vit else None
        model, phi = mllm.build_stage2(cfg, vit_dir)
        (out / "vit_from.txt").write_text(str(vit_dir or ""))
    img_tok = mllm.image_token_id(model)
    params = [p for p in model.parameters() if p.requires_grad] + (list(phi.parameters()) if phi else [])

    import bitsandbytes as bnb
    opt = (bnb.optim.PagedAdamW8bit if T.optim == "paged_adamw_8bit" else torch.optim.AdamW)(
        params, lr=T.lr, betas=tuple(T.betas), weight_decay=0.0)
    steps_per_epoch = math.ceil(len(samples) / (T.batch_size * T.grad_accum))
    total = steps_per_epoch * epochs
    sched = cosine_with_warmup(opt, total, int(T.warmup_ratio * total))

    # phi is checkpointed alongside the model through a thin container
    holder = torch.nn.ModuleDict({"model": model, **({"phi": phi} if phi else {})})
    ck = Checkpointer(out, T.save_every)
    state = ck.resume(holder, opt, sched)
    log = JsonlLog(out / "log.jsonl")
    collate = Collator(proc, T.max_len)
    feats = cfg.work / "feats" if use_fgra else None
    w = cfg.stage2.fgra_weight
    print(json.dumps({"stage": a.stage, "name": a.name, "samples": len(samples), "epochs": epochs,
                      "opt_steps": total, "fgra": use_fgra, "vit_from": a.vit}))

    step, model_ = state["step"], model
    model.train()
    for epoch in range(state["epoch"], epochs):
        g = torch.Generator().manual_seed(cfg.seed + epoch)
        order = torch.randperm(len(samples), generator=g).tolist()
        skip = state["micro"] if epoch == state["epoch"] else 0
        ds = SFTDataset(cfg, [samples[i] for i in order[skip:]], feats)
        dl = DataLoader(ds, batch_size=T.batch_size, num_workers=T.num_workers, collate_fn=collate)
        micro, t0, acc = skip, time.time(), {"lm": 0.0, "fgra": 0.0, "cos": 0.0, "n": 0}
        for batch in dl:
            feat = batch.pop("feat", None)
            batch.pop("kinds")
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model_(**batch, output_hidden_states=use_fgra)
                loss = o.loss
                if use_fgra:
                    l_fgra, cos = mllm.fgra_loss(phi, o.hidden_states[cfg.stage2.fgra_layer],
                                                 batch["input_ids"], img_tok, feat.cuda())
                    loss = loss + w * l_fgra
                    acc["fgra"] += l_fgra.item(); acc["cos"] += cos.item()
            (loss / T.grad_accum).backward()
            acc["lm"] += o.loss.item(); acc["n"] += 1
            micro += batch["input_ids"].shape[0]
            if micro % (T.batch_size * T.grad_accum) == 0 or micro == len(samples):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % T.log_every == 0 or step == 1:
                    n = max(1, acc["n"])
                    log.write(step=step, epoch=epoch, of=total, l_lm=acc["lm"] / n,
                              **({"l_fgra": acc["fgra"] / n, "cos": acc["cos"] / n} if use_fgra else {}),
                              lr=sched.get_last_lr()[0], s_per_sample=(time.time() - t0) / max(1, micro - skip),
                              peak_gb=torch.cuda.max_memory_allocated() / 2**30)
                    acc = {"lm": 0.0, "fgra": 0.0, "cos": 0.0, "n": 0}
                ck.maybe_save(step, holder, opt, sched, epoch=epoch, micro=micro)
                if a.max_steps and step >= a.max_steps:
                    break
        if a.max_steps and step >= a.max_steps:
            break
        ck.save(step, holder, opt, sched, epoch=epoch + 1, micro=0)

    if a.stage == 1:
        mllm.save_stage1(model, out)
    else:
        mllm.save_stage2(model, phi, out)
    (out / "DONE").write_text(f"{step} steps\n")
    print("saved final weights to", out)


if __name__ == "__main__":
    main()
