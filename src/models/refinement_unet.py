"""Tiny UNet that refines a PatchCore score map into a defect mask.

Input  : 2-channel tensor (B, 2, H, W)
         channel 0 = PatchCore score map, divided by train_pixel_max
         channel 1 = grayscale of original image, divided by 255
Output : 1-channel sigmoid probability map (B, 1, H, W) -- defect = 1

Architecture: 4-level encoder/decoder with skip connections, base channel 16.
~80k parameters, deliberately tiny because the supervised dataset is
~10 images. Dropout 0.2 in the bottleneck for regularisation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch: int, out_ch: int, dropout: float = 0.0):
    layers = [
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


class RefinementUNet(nn.Module):
    def __init__(self, in_ch: int = 2, base: int = 16, dropout: float = 0.2):
        super().__init__()
        self.enc1 = conv_block(in_ch, base)
        self.enc2 = conv_block(base, base * 2)
        self.enc3 = conv_block(base * 2, base * 4)
        self.enc4 = conv_block(base * 4, base * 8, dropout=dropout)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)        # logits; apply sigmoid externally


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor,
                  dice_w: float = 0.5, bce_w: float = 0.5,
                  pos_weight: float | None = None) -> torch.Tensor:
    """Dice for shape, BCE-with-logits for per-pixel gradient. pos_weight
    rebalances when defect pixels are <<1% of the image (use ~50-200)."""
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(2, 3))
    denom = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1e-6
    dice = 1 - (2 * inter + 1e-6) / denom
    if pos_weight is not None:
        bce = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=torch.tensor(pos_weight, device=logits.device))
    else:
        bce = F.binary_cross_entropy_with_logits(logits, target)
    return dice_w * dice.mean() + bce_w * bce
