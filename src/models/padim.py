"""PaDiM (Defard et al., 2020): per-location multivariate Gaussian over
backbone-feature embeddings, scored by Mahalanobis distance.

Same interface as PatchCoreOfficial:
    fit(dataloader)                          -> trains the per-location stats
    predict(images, target_size)             -> (score_maps_low, image_scores)
    save(path) / load(path)

Where PatchCore picks K-NN distance to a memory bank, PaDiM picks the
Mahalanobis distance from a Gaussian fit at each (h, w) cell. PaDiM
assumes the training set is in roughly the same canonical pose -- which
matches the Foosung line-scan setting (each frame captures the part at
the same orientation). For free-pose categories (hazelnut etc.) PaDiM
falls behind PatchCore.

A fixed-seed random projection (orig_dim -> proj_dim, default 100) shrinks
the per-location covariance matrices to manageable size; matches the
paper. Tikhonov regularisation (eps * I) on the covariance prevents the
inverse from blowing up when only a few train samples are available.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.models.feature_extractor import build_feature_extractor


class PaDiM:
    def __init__(self,
                 backbone: str = 'wide_resnet50_2',
                 layers: Sequence = ('layer1', 'layer2', 'layer3'),
                 input_size: int = 224,
                 proj_dim: int = 100,
                 cov_eps: float = 0.01,
                 device: str = 'cuda'):
        self.backbone_name = backbone
        self.layers = tuple(layers)
        self.input_size = int(input_size)
        self.proj_dim = int(proj_dim)
        self.cov_eps = float(cov_eps)
        self.device = device

        self.feature_embedder = build_feature_extractor(
            backbone=backbone, layers=layers,
        ).to(device).eval()

        # State filled by fit() / load(): per-position mean (HW, D) and
        # inverse covariance (HW, D, D), plus the (orig_dim, proj_dim)
        # random projection matrix.
        self._means: Optional[torch.Tensor] = None
        self._covs_inv: Optional[torch.Tensor] = None
        self._proj_mat: Optional[torch.Tensor] = None
        self.feature_map_shape: Optional[Tuple[int, int, int]] = None  # (D, H, W)

    @torch.no_grad()
    def _embed_raw(self, images: torch.Tensor) -> torch.Tensor:
        """Concatenate backbone features at all configured layers, aligned
        to the first layer's spatial resolution. Returns (B, C_total, H, W)."""
        feats = self.feature_embedder(images)
        ordered = list(feats.values())
        target_h, target_w = ordered[0].shape[-2:]
        aligned = []
        for f in ordered:
            if f.shape[-2:] != (target_h, target_w):
                f = F.interpolate(f, size=(target_h, target_w),
                                  mode='bilinear', align_corners=False)
            aligned.append(f)
        return torch.cat(aligned, dim=1)

    def _init_proj(self, orig_dim: int):
        """Fixed-seed random projection (Gaussian columns, unit norm)."""
        g = torch.Generator(device='cpu').manual_seed(42)
        proj = torch.randn(orig_dim, self.proj_dim, generator=g)
        proj = proj / proj.norm(dim=0, keepdim=True)
        self._proj_mat = proj.to(self.device)

    @torch.no_grad()
    def _project(self, raw: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, H, W, proj_dim)."""
        B, C, H, W = raw.shape
        if self._proj_mat is None:
            self._init_proj(C)
        # Move C to last dim, then matmul with projection.
        x = raw.permute(0, 2, 3, 1).reshape(-1, C)
        x = x @ self._proj_mat
        return x.reshape(B, H, W, self.proj_dim)

    @torch.no_grad()
    def fit(self, dataloader) -> None:
        """Collect projected features from every train image, then estimate
        a multivariate Gaussian per (h, w) cell."""
        all_feats = []
        for batch in tqdm(dataloader, desc='PaDiM: collecting features'):
            images = batch['image'].to(self.device, non_blocking=True)
            raw = self._embed_raw(images)               # (B, C, H, W)
            B, C, H, W = raw.shape
            self.feature_map_shape = (self.proj_dim, H, W)
            f_proj = self._project(raw)                  # (B, H, W, D)
            all_feats.append(f_proj.cpu())
        all_feats = torch.cat(all_feats, dim=0)         # (N, H, W, D)
        N, H, W, D = all_feats.shape
        print(f'PaDiM fit: N={N}, fm={H}x{W}, proj_dim={D}')

        # Reshape to (HW, N, D)
        flat = all_feats.permute(1, 2, 0, 3).reshape(H * W, N, D).to(self.device)
        means = flat.mean(dim=1)                         # (HW, D)
        centered = flat - means.unsqueeze(1)             # (HW, N, D)
        # Covariance per position: (HW, D, D)
        denom = max(1, N - 1)
        covs = torch.einsum('phd,phe->pde', centered, centered) / denom
        eye = torch.eye(D, device=self.device).unsqueeze(0)
        covs = covs + self.cov_eps * eye
        covs_inv = torch.linalg.inv(covs)

        self._means = means.contiguous()                 # (HW, D)
        self._covs_inv = covs_inv.contiguous()           # (HW, D, D)
        print(f'PaDiM stats: means {tuple(self._means.shape)}, '
              f'covs_inv {tuple(self._covs_inv.shape)}')

    @torch.no_grad()
    def predict(self, images: torch.Tensor,
                target_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Return (score_maps, image_scores). Score maps are bilinearly
        upsampled to target_size; image_scores = max over each map."""
        if self._means is None or self._covs_inv is None:
            raise RuntimeError('PaDiM.fit() must be called before predict().')

        raw = self._embed_raw(images)
        f_proj = self._project(raw)                      # (B, H, W, D)
        B, H, W, D = f_proj.shape
        # (HW, B, D)
        q = f_proj.permute(1, 2, 0, 3).reshape(H * W, B, D)
        diff = q - self._means.unsqueeze(1)              # (HW, B, D)
        # Mahalanobis: d^2 = diff^T @ Σ⁻¹ @ diff per position, batched
        # einsum: (HW, B, D) x (HW, D, D) -> (HW, B, D), then sum-dot diff -> (HW, B)
        proj = torch.einsum('pbd,pde->pbe', diff, self._covs_inv)
        maha2 = (proj * diff).sum(dim=-1)               # (HW, B)
        maha = torch.sqrt(torch.clamp(maha2, min=0.0))   # (HW, B)
        score_maps_low = maha.t().reshape(B, 1, H, W)     # (B, 1, H, W)
        image_scores = score_maps_low.view(B, -1).max(dim=1).values

        score_maps = F.interpolate(
            score_maps_low.float(),
            size=target_size,
            mode='bilinear',
            align_corners=False,
        )[:, 0].cpu().numpy()
        return score_maps, image_scores.cpu().numpy()

    # ---- io --------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'means': self._means.cpu() if self._means is not None else None,
            'covs_inv': self._covs_inv.cpu() if self._covs_inv is not None else None,
            'proj_mat': self._proj_mat.cpu() if self._proj_mat is not None else None,
            'feature_map_shape': self.feature_map_shape,
            'backbone': self.backbone_name,
            'layers': self.layers,
            'input_size': self.input_size,
            'proj_dim': self.proj_dim,
            'cov_eps': self.cov_eps,
        }, path)

    def load(self, path: str) -> None:
        blob = torch.load(path, map_location='cpu', weights_only=False)
        self._means = blob['means'].to(self.device) if blob['means'] is not None else None
        self._covs_inv = blob['covs_inv'].to(self.device) if blob['covs_inv'] is not None else None
        self._proj_mat = blob['proj_mat'].to(self.device) if blob['proj_mat'] is not None else None
        self.feature_map_shape = blob.get('feature_map_shape')
        self.backbone_name = blob.get('backbone', self.backbone_name)
        self.layers = tuple(blob.get('layers', self.layers))
        self.input_size = int(blob.get('input_size', self.input_size))
        self.proj_dim = int(blob.get('proj_dim', self.proj_dim))
        self.cov_eps = float(blob.get('cov_eps', self.cov_eps))
