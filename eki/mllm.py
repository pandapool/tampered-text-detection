"""Qwen2.5-VL loading for each stage: NF4 LLM + bf16 ViT, LoRA placement, FGRA head and loss.

Stage 1 (paper IV-B):  ViT LoRA + projector trainable, LLM frozen (gradients still flow through it).
Stage 2 (paper IV-C):  ViT frozen, LLM LoRA + projector + phi trainable, loss = L_LM + w * L_FGRA.
Baseline / M_pre:      Stage 2 from the original ViT with w = 0 (see DECISIONS.md).
"""
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from eki.train_utils import count_trainable

EXPERT_DIM = 1024  # DINOv2-ViT-L hidden size


# ---------------------------------------------------------------- model parts (version-robust)

def unwrap(model):
    return model.get_base_model() if isinstance(model, PeftModel) else model


def visual(model):
    m = unwrap(model)
    return m.visual if hasattr(m, "visual") else m.model.visual


def merger(model):
    return visual(model).merger


def text_config(model):
    c = unwrap(model).config
    return getattr(c, "text_config", c)


def image_token_id(model):
    return unwrap(model).config.image_token_id


# ---------------------------------------------------------------- loading

def processor(cfg):
    px = cfg.data.img_size ** 2  # pin the resize so every image gives the same 36x36 patch grid
    return AutoProcessor.from_pretrained(cfg.model.mllm, min_pixels=px, max_pixels=px)


def load_base(cfg):
    dtype = getattr(torch, cfg.model.compute_dtype)
    q = None
    if cfg.model.load_4bit:
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
                               # transformers 5 matches these as prefixes of the full module name
                               llm_int8_skip_modules=["model.visual", "visual", "lm_head"])
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.model.mllm, dtype=dtype, quantization_config=q,
        attn_implementation=cfg.model.attn_implementation, device_map={"": 0})
    model.config.use_cache = False
    quantised = [n for n, m in visual(model).named_modules() if "4bit" in type(m).__name__]
    assert not quantised, f"ViT must stay bf16, but these were quantised: {quantised[:3]}"
    return model


def load_vit(model, path):
    """Load a merged Stage 1 ViT (+ projector) saved by save_stage1."""
    sd = torch.load(Path(path) / "visual.pt", map_location="cpu")
    visual(model).load_state_dict(sd, strict=True)


def _regex(names, scope):
    alt = "|".join(re.escape(n) for n in names)
    return rf".*{scope}\.\d+\.({alt})$"


def prep_for_training(model):
    """Freeze everything and enable non-reentrant gradient checkpointing (LLM and ViT).

    Replaces PEFT's prepare_model_for_kbit_training, which would upcast the bf16 ViT to fp32
    (+1.3 GB) and freeze parameters we re-enable later.
    """
    for p in model.parameters():
        p.requires_grad = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False


def _check_targets(model, targets):
    """Every configured LoRA target name must have received an adapter."""
    hit = {t for t in targets for n, _ in model.named_modules() if n.endswith(t + ".lora_A")}
    missing = set(targets) - hit
    assert not missing, f"LoRA targets matched no module: {sorted(missing)}"


def _train_projector(model):
    for p in merger(model).parameters():
        p.data = p.data.float()  # fp32 master weights; forward runs under bf16 autocast
        p.requires_grad = True


def build_stage1(cfg):
    model = load_base(cfg)
    prep_for_training(model)
    lc = LoraConfig(r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout, bias="none",
                    target_modules=_regex(cfg.lora.vit_targets, r"visual\.blocks"))
    model = get_peft_model(model, lc)
    _train_projector(model)
    _check_targets(model, cfg.lora.vit_targets)
    assert not any(p.requires_grad for n, p in model.named_parameters()
                   if "visual" not in n), "Stage 1 must leave the LLM frozen"
    count_trainable(model)
    return model


class Phi(nn.Module):
    """FGRA projection head: LLM hidden size -> expert feature size."""

    def __init__(self, d_in, d_out, kind="mlp"):
        super().__init__()
        self.net = nn.Linear(d_in, d_out) if kind == "linear" else \
            nn.Sequential(nn.Linear(d_in, d_in), nn.GELU(), nn.Linear(d_in, d_out))

    def forward(self, x):
        return self.net(x)


def build_stage2(cfg, vit_from=None):
    model = load_base(cfg)
    if vit_from:
        load_vit(model, vit_from)
        print(f"loaded Stage 1 ViT + projector from {vit_from}")
    prep_for_training(model)
    lc = LoraConfig(r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout, bias="none",
                    target_modules=_regex(cfg.lora.llm_targets, r"layers"))
    model = get_peft_model(model, lc)
    _check_targets(model, cfg.lora.llm_targets)
    _train_projector(model)
    assert not any(p.requires_grad for n, p in model.named_parameters()
                   if "visual" in n and "merger" not in n), "Stage 2 must leave the ViT frozen"
    phi = None
    if cfg.stage2.use_fgra:
        phi = Phi(text_config(model).hidden_size, EXPERT_DIM, cfg.stage2.phi).cuda()
    count_trainable(model)
    return model, phi


# ---------------------------------------------------------------- saving / loading results

def save_stage1(model, out):
    """Merge the ViT LoRA into the bf16 ViT weights and save ViT + projector."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    base = model.merge_and_unload()
    sd = {k: v.to(torch.bfloat16).cpu() for k, v in visual(base).state_dict().items()}
    torch.save(sd, out / "visual.pt")
    return base


def save_stage2(model, phi, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "adapter")
    torch.save({k: v.to(torch.bfloat16).cpu() for k, v in merger(model).state_dict().items()}, out / "merger.pt")
    if phi is not None:
        torch.save(phi.state_dict(), out / "phi.pt")


def load_for_inference(cfg, run_dir):
    """run_dir holds visual.pt (Stage 1 output) and/or adapter/ + merger.pt (Stage 2 output).

    A Stage 2 run records which ViT it started from in vit_from.txt.
    """
    run_dir = Path(run_dir)
    model = load_base(cfg)
    vit_src = run_dir if (run_dir / "visual.pt").exists() else None
    if (run_dir / "vit_from.txt").exists():
        src = (run_dir / "vit_from.txt").read_text().strip()
        vit_src = Path(src) if src else None
    if vit_src:
        load_vit(model, vit_src)
    if (run_dir / "adapter").exists():
        merger(model).load_state_dict(torch.load(run_dir / "merger.pt", map_location="cpu"))
        model = PeftModel.from_pretrained(model, run_dir / "adapter")
    model.eval()
    model.config.use_cache = True
    return model


# ---------------------------------------------------------------- FGRA

def visual_hidden(hidden, input_ids, img_tok):
    """[B, L, D] hidden states -> [B, K, D] at the image-token positions (K = 324 at 504 px)."""
    m = input_ids == img_tok
    k = int(m[0].sum())
    assert bool((m.sum(1) == k).all()), "every image in the batch must yield the same token count"
    return hidden[m].view(hidden.shape[0], k, -1)


def fgra_loss(phi, hidden, input_ids, img_tok, feat):
    """Eq. 3: -mean_i cos(phi(h_i), f_exp,i). Returns (loss, mean cosine)."""
    h = visual_hidden(hidden, input_ids, img_tok)
    assert h.shape[1] == feat.shape[1], f"token grid {h.shape[1]} != expert grid {feat.shape[1]}"
    cos = F.cosine_similarity(phi(h.float()), feat.float(), dim=-1)
    return -cos.mean(), cos.mean().detach()
