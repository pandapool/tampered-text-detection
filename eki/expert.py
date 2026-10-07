"""Forensic expert (paper Sec. IV-C1): DINOv2-ViT-L/14 backbone + segmentation head.

The paper uses Mask2Former; the head is discarded before Stage 2, so a light conv/upsample head
is used here. Only the backbone's last-layer patch tokens are kept as the FGRA teacher.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
PATCH = 14


def to_tensor(imgs):
    """List of PIL images -> normalised float tensor [B, 3, H, W]."""
    import numpy as np

    x = torch.from_numpy(np.stack([np.asarray(i, dtype=np.float32) / 255.0 for i in imgs]))
    return (x.permute(0, 3, 1, 2) - MEAN) / STD


class Head(nn.Module):
    def __init__(self, dim=1024, ch=256):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(dim, ch, 1), nn.GroupNorm(32, ch), nn.GELU())
        self.up = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(32, ch), nn.GELU()),
            nn.Sequential(nn.Conv2d(ch, ch // 2, 3, padding=1), nn.GroupNorm(16, ch // 2), nn.GELU()),
        ])
        self.out = nn.Conv2d(ch // 2, 1, 1)

    def forward(self, grid):  # [B, D, g, g] -> logits [B, 1, 4g, 4g]
        x = self.proj(grid)
        for blk in self.up:
            x = blk(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        return self.out(x)


class Expert(nn.Module):
    def __init__(self, name="facebook/dinov2-large", freeze_blocks=12, head_ch=256):
        super().__init__()
        self.backbone = Dinov2Model.from_pretrained(name)
        self.head = Head(self.backbone.config.hidden_size, head_ch)
        self.freeze(freeze_blocks)

    def freeze(self, n):
        for p in self.backbone.embeddings.parameters():
            p.requires_grad = False
        for blk in self.backbone.encoder.layer[:n]:
            for p in blk.parameters():
                p.requires_grad = False

    def patch_tokens(self, x):
        """[B, 3, H, W] -> last-layer patch tokens after the final norm, [B, g, g, D] (CLS dropped)."""
        h = self.backbone(pixel_values=x).last_hidden_state
        n_reg = getattr(self.backbone.config, "num_register_tokens", 0)
        tok = h[:, 1 + n_reg:]
        g = x.shape[-1] // PATCH
        return tok.reshape(x.shape[0], g, g, -1)

    def forward(self, x):
        tok = self.patch_tokens(x)
        logits = self.head(tok.permute(0, 3, 1, 2))
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def bce_dice(logits, target, eps=1.0):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    dice = 1 - (2 * (p * t).sum(1) + eps) / (p.sum(1) + t.sum(1) + eps)
    return bce + dice.mean()


@torch.no_grad()
def fgra_targets(expert, x, merge=2):
    """Teacher features matched to Qwen's merged token grid: 36x36 -> 2x2 avg pool -> [B, 324, D]."""
    tok = expert.patch_tokens(x).permute(0, 3, 1, 2)
    pooled = F.avg_pool2d(tok.float(), merge)
    return pooled.flatten(2).transpose(1, 2)  # row-major over the merged grid
