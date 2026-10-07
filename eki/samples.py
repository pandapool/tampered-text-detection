"""Instruction samples (global / TF / IF) and the chat-format collator with answer-only labels.

A sample is a dict: {"kind": "global"|"tf"|"if_", "img": <sample id or patch path>,
"prompt": str, "answer": str, "id": <sample id the image came from>}.
"""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from eki import boxes as B
from eki.data import Splits, read_jsonl

IGNORE = -100


def global_samples(cfg, ids, boxes_by_id):
    return [{"kind": "global", "img": i, "id": i, "prompt": cfg.prompts["global"],
             "answer": B.format_answer(boxes_by_id[i])} for i in ids]


def tf_samples(cfg):
    return [{"kind": "tf", "img": r["id"], "id": r["id"],
             "prompt": cfg.prompts.tf.format(text=r["text"], box=B.fmt_box(r["box"])),
             "answer": B.format_answer(r["gt"])}
            for r in read_jsonl(cfg.work / "tf_pairs.jsonl")]


def if_samples(cfg):
    return [{"kind": "if_", "img": str(cfg.work / r["path"]), "id": r["id"], "prompt": cfg.prompts.if_,
             "answer": "Tampered." if r["tampered"] else B.REAL}
            for r in read_jsonl(cfg.work / "patches" / "index.jsonl")]


def mix(sources, weights, cap, seed):
    """Mixture of sample sources anchored on the global set, without repetition.

    Source k contributes min(len(k), weight_k * len(global)) samples, so equal weights give every
    source the same share as global localisation (the roadmap's "equal sampling weight per
    source"). If the total exceeds `cap`, every source is scaled down proportionally.
    """
    rng = random.Random(seed)
    anchor = len(sources.get("global", [])) or max(len(v) for v in sources.values())
    n = {k: min(len(v), round(anchor * weights.get(k, 1.0))) for k, v in sources.items() if v}
    total = sum(n.values())
    if total > cap:
        n = {k: round(c * cap / total) for k, c in n.items()}
    out = [x for k, c in n.items() for x in rng.sample(sources[k], c)]
    rng.shuffle(out)
    return out


class SFTDataset(Dataset):
    def __init__(self, cfg, samples, feats_dir=None):
        self.cfg, self.samples, self.S = cfg, samples, Splits(cfg)
        self.size, self.feats_dir = cfg.data.img_size, feats_dir

    def __len__(self):
        return len(self.samples)

    def load_image(self, s):
        if s["kind"] == "if_":
            return Image.open(s["img"]).convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        return self.S.image(s["img"])

    def __getitem__(self, i):
        s = self.samples[i]
        out = {**s, "image": self.load_image(s)}
        if self.feats_dir is not None and s["kind"] == "global":
            out["feat"] = torch.from_numpy(np.load(Path(self.feats_dir) / f"{s['id'].replace(':', '_')}.npy"))
        return out


def messages(prompt):
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]


class Collator:
    """Tokenises prompt and answer separately so labels cover only answer tokens + <|im_end|>."""

    def __init__(self, processor, max_len):
        self.p, self.max_len = processor, max_len
        self.tok = processor.tokenizer
        self.eos = self.tok.convert_tokens_to_ids("<|im_end|>")

    def prompt_text(self, prompt):
        return self.p.apply_chat_template(messages(prompt), tokenize=False, add_generation_prompt=True)

    def __call__(self, batch):
        texts = [self.prompt_text(b["prompt"]) for b in batch]
        enc = self.p(text=texts, images=[b["image"] for b in batch], return_tensors="pt", padding=True)
        # per-token side channels (transformers 5: mm_token_type_ids feeds the 3D-RoPE index)
        extra = [k for k, v in enc.items() if k not in ("input_ids", "attention_mask") and torch.is_tensor(v)
                 and v.shape[:2] == enc["input_ids"].shape]
        rows = []
        for k, b in enumerate(batch):
            keep = enc["attention_mask"][k].bool()
            p_ids = enc["input_ids"][k][keep]
            a_ids = torch.tensor(self.tok(b["answer"], add_special_tokens=False)["input_ids"] + [self.eos])
            r = {"input_ids": torch.cat([p_ids, a_ids]),
                 "labels": torch.cat([torch.full_like(p_ids, IGNORE), a_ids])}
            for e in extra:  # answer tokens are text: type 0
                r[e] = torch.cat([enc[e][k][keep], torch.zeros_like(a_ids, dtype=enc[e].dtype)])
            rows.append({n: v[: self.max_len] for n, v in r.items()})
        L = max(len(r["input_ids"]) for r in rows)
        fill = {"input_ids": self.tok.pad_token_id, "labels": IGNORE}
        out = {n: torch.full((len(rows), L), fill.get(n, 0), dtype=rows[0][n].dtype) for n in rows[0]}
        out["attention_mask"] = torch.zeros((len(rows), L), dtype=torch.long)
        for k, r in enumerate(rows):  # right padding for training
            for n, v in r.items():
                out[n][k, : len(v)] = v
            out["attention_mask"][k, : len(r["input_ids"])] = 1
        out.update(pixel_values=enc["pixel_values"], image_grid_thw=enc["image_grid_thw"],
                   kinds=[b["kind"] for b in batch])
        if any("feat" in b for b in batch):
            assert all("feat" in b for b in batch), "FGRA batches must be global-only"
            out["feat"] = torch.stack([b["feat"] for b in batch])
        return out
