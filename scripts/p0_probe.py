"""Phase 0: environment and memory probe.

  python scripts/p0_probe.py            # all checks
  python scripts/p0_probe.py --only order stage1

  order   processor gives a 36x36 patch grid -> 324 merged tokens, in row-major order over the
          18x18 merged grid (so token i pairs with pooled DINOv2 cell i)
  stage1  fwd + bwd with ViT LoRA + projector trainable, gradients through the frozen NF4 LLM
  stage2  fwd + bwd with LLM LoRA + projector + phi, all hidden states returned, FGRA loss on
  expert  fwd + bwd of DINOv2-L (last 12 blocks) + head at 504 px, batch 2
Gate: every training peak < 7.2 GB. Writes p0_report.json with peaks and seconds per sample.
"""
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from eki import boxes as B
from eki import mllm
from eki.config import cli
from eki.samples import Collator

sys.path.insert(0, str(Path(__file__).parent))
from make_synthetic import fonts, page  # noqa: E402

GATE_GB = 7.2


def sample_image(size, seed=0):
    img, mask = page(random.Random(seed), fonts(), True)
    img = img.resize((size, size), Image.BICUBIC)
    mask = np.array(Image.fromarray(mask).resize((size, size), Image.NEAREST)) > 0
    return img, B.format_answer(B.mask_to_boxes(mask))


def check_order(cfg):
    proc = mllm.processor(cfg)
    size, cell = cfg.data.img_size, 28  # 28 px = one merged token (2x2 patches of 14 px)
    g = size // cell
    arr = np.zeros((size, size, 3), np.uint8)
    for r in range(g):
        for c in range(g):
            arr[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = (r * 13 % 256, c * 13 % 256, 128)
    enc = proc(text=["<|vision_start|><|image_pad|><|vision_end|>"], images=[Image.fromarray(arr)], return_tensors="pt")
    thw = enc["image_grid_thw"][0].tolist()
    n_tok = int((enc["input_ids"] == proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")).sum())
    pv = enc["pixel_values"].float()  # [t*h*w, C*T*14*14], patch order grouped by 2x2 merge window
    per_patch = pv.view(pv.shape[0], 3, -1).mean(-1)  # normalised channel means
    merged = per_patch.view(-1, 4, 3).mean(1)          # one row per merged token
    # recover (r, c) from the R and G channel ranks
    r_rank = torch.unique(merged[:, 0], return_inverse=True)[1].view(g, g)
    c_rank = torch.unique(merged[:, 1], return_inverse=True)[1].view(g, g)
    row_major = bool((r_rank == torch.arange(g)[:, None]).all() and (c_rank == torch.arange(g)[None, :]).all())
    res = {"grid_thw": thw, "image_tokens": n_tok, "row_major": row_major}
    print("order:", res)
    assert thw == [1, size // 14, size // 14], f"processor resized the image: {thw}"
    assert n_tok == (size // 28) ** 2, n_tok
    assert row_major, "merged-token order is not row-major; FGRA pairing would be wrong"
    return res


def _measure(fn, n=3):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(n):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
    return {"peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
            "s_per_sample": round(float(np.median(times[1:])), 3)}


def _batch(cfg, proc):
    img, ans = sample_image(cfg.data.img_size)
    b = Collator(proc, cfg.train.max_len)([{"kind": "global", "image": img, "prompt": cfg.prompts["global"],
                                             "answer": ans}])
    b.pop("kinds")
    print(f"probe answer ({int((b['labels'] != -100).sum())} tokens): {ans[:80]}")
    return {k: v.cuda() for k, v in b.items()}


def probe_stage1(cfg):
    proc = mllm.processor(cfg)
    model = mllm.build_stage1(cfg); model.train()
    batch = _batch(cfg, proc)
    w_gb = torch.cuda.memory_allocated() / 2**30

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss
        loss.backward()
        vit_grad = sum(p.grad.abs().sum().item() for n, p in model.named_parameters()
                       if p.requires_grad and "lora_" in n and p.grad is not None)
        assert vit_grad > 0, "no gradient reached the ViT LoRA"
        model.zero_grad(set_to_none=True)

    r = {"weights_gb": round(w_gb, 2), **_measure(step)}
    del model; gc.collect(); torch.cuda.empty_cache()
    return r


def probe_stage2(cfg):
    proc = mllm.processor(cfg)
    model, phi = mllm.build_stage2(cfg); model.train()
    batch = _batch(cfg, proc)
    feat = torch.randn(1, 324, mllm.EXPERT_DIM, device="cuda")
    tok = mllm.image_token_id(model)
    w_gb = torch.cuda.memory_allocated() / 2**30

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(**batch, output_hidden_states=True)
            l, _ = mllm.fgra_loss(phi, o.hidden_states[cfg.stage2.fgra_layer], batch["input_ids"], tok, feat)
            (o.loss + l).backward()
        model.zero_grad(set_to_none=True); phi.zero_grad(set_to_none=True)

    r = {"weights_gb": round(w_gb, 2), **_measure(step)}
    del model, phi; gc.collect(); torch.cuda.empty_cache()
    return r


def probe_expert(cfg):
    from eki.expert import Expert, bce_dice
    E = cfg.expert
    m = Expert(cfg.model.expert, E.freeze_blocks, E.head_channels).cuda().train()
    m.backbone.gradient_checkpointing_enable()
    s = cfg.data.img_size
    x = torch.randn(E.batch_size, 3, s, s, device="cuda")
    y = (torch.rand(E.batch_size, 1, s, s, device="cuda") > 0.99).float()

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(x)
        bce_dice(out.float(), y).backward()
        m.zero_grad(set_to_none=True)

    r = _measure(step)
    r["s_per_sample"] = round(r["s_per_sample"] / E.batch_size, 3)
    del m; gc.collect(); torch.cuda.empty_cache()
    return r


def main():
    cfg, a = cli(__doc__, lambda p: p.add_argument("--only", nargs="*",
                                                   default=["order", "stage1", "stage2", "expert"]))
    print(f"GPU: {torch.cuda.get_device_name()}, bf16: {torch.cuda.is_bf16_supported()}, "
          f"free: {torch.cuda.mem_get_info()[0] / 2**30:.2f} GB")
    path = cfg.work / "p0_report.json"
    rep = json.loads(path.read_text()) if path.exists() else {}
    fns = {"order": check_order, "stage1": probe_stage1, "stage2": probe_stage2, "expert": probe_expert}
    for k in a.only:
        rep[k] = fns[k](cfg)
        print(k, rep[k], flush=True)
        path.write_text(json.dumps(rep, indent=2))
    for k in ("stage1", "stage2", "expert"):
        if k in rep:
            ok = rep[k]["peak_gb"] < GATE_GB
            print(f"gate {k}: peak {rep[k]['peak_gb']} GB {'< ' if ok else '>= '}{GATE_GB}  {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
