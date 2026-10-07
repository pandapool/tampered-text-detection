"""Step-level checkpoint/resume and jsonl logging shared by all training scripts.

Only trainable parameters are checkpointed (frozen weights are reloaded from their source), so a
checkpoint stays small. Data order is a seeded permutation per epoch, so resuming only needs the
epoch and the number of samples already consumed (``micro``).
"""
import json
import os
import time
from pathlib import Path

import torch


def trainable_state(model):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in names}


def count_trainable(model, verbose=True):
    n_t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"trainable params: {n_t / 1e6:.2f}M / {n_all / 1e6:.1f}M")
    return n_t


class Checkpointer:
    def __init__(self, out, every):
        self.out, self.every = Path(out), every
        self.out.mkdir(parents=True, exist_ok=True)
        self.path = self.out / "ckpt.pt"

    def save(self, step, model, opt, sched, **extra):
        tmp = self.path.with_suffix(".tmp")
        torch.save({"step": step, "model": trainable_state(model), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "rng": torch.get_rng_state(), **extra}, tmp)
        os.replace(tmp, self.path)

    def maybe_save(self, step, *a, **kw):
        if self.every and step % self.every == 0:
            self.save(step, *a, **kw)

    def resume(self, model, opt, sched):
        if not self.path.exists():
            return {"step": 0, "epoch": 0, "micro": 0}
        ck = torch.load(self.path, map_location="cpu", weights_only=False)
        missing = set(trainable_state(model)) - set(ck["model"])
        assert not missing, f"checkpoint lacks trainable params: {sorted(missing)[:5]}"
        model.load_state_dict(ck["model"], strict=False)
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        torch.set_rng_state(ck["rng"])
        print(f"resumed from {self.path} at step {ck['step']} (epoch {ck['epoch']}, sample {ck['micro']})")
        return ck


class JsonlLog:
    def __init__(self, path):
        self.path = Path(path)

    def write(self, **kv):
        kv["time"] = time.time()
        line = json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in kv.items()})
        with open(self.path, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)


def cosine_with_warmup(opt, total, warmup):
    import math

    def f(s):
        if s < warmup:
            return (s + 1) / max(1, warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warmup) / max(1, total - warmup))))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)
