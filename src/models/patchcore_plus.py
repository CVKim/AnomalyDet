"""PatchCore + post-paper extensions that are commonly used in practice
but NOT in the paper-faithful default. Each extension is opt-in via the
`extensions` constructor arg (list of strings) so they can be ablated.

Extensions:
  - `position_features`  : concat normalised (x, y) coordinates to every
                            patch embedding before coreset / NN scoring.
                            Helps when defects have a position prior
                            (e.g. cracks always near the part edge).
  - `softmax_reweight`   : Roth 2022 §3.4 — instead of K-mean NN distance,
                            use the K nearest distances softmaxed and
                            weight by the inverse of the K-th nearest
                            distance (boosts isolated coreset hits).
  - `multi_scale`         : average feature maps from two resolutions
                            (`input_size` and `input_size // 2`) before
                            patch aggregation. Cheap precursor to the
                            multi-scale ensemble runner.

The plain `PatchCoreOfficial` is unchanged; this class subclasses it
and overrides only what each extension touches. Defaults give identical
behaviour to the parent.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.models.patchcore_official import PatchCoreOfficial


class PatchCorePlus(PatchCoreOfficial):
    def __init__(self,
                 extensions: Optional[List[str]] = None,
                 position_weight: float = 0.1,
                 reweight_k: int = 9,
                 **kwargs):
        # PatchCoreOfficial accepts fp16; pass through transparently.
        super().__init__(**kwargs)
        self.extensions = list(extensions or [])
        self.position_weight = float(position_weight)
        self.reweight_k = int(reweight_k)
        # Pre-built position grid; lazily filled in on first call.
        self._pos_grid: Optional[torch.Tensor] = None
        self._pos_dim: int = 0

    def _build_pos_grid(self, H: int, W: int) -> torch.Tensor:
        """Returns (1, 2, H, W) tensor with normalised (x, y) in [-1, 1],
        scaled by self.position_weight so the magnitude matches the
        feature norm rather than dominating it."""
        ys = torch.linspace(-1.0, 1.0, H, device=self.device)
        xs = torch.linspace(-1.0, 1.0, W, device=self.device)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        pos = torch.stack([gx, gy], dim=0).unsqueeze(0) * self.position_weight
        self._pos_grid = pos
        self._pos_dim = 2
        return pos

    @torch.no_grad()
    def _embed(self, images: torch.Tensor) -> torch.Tensor:
        """Same as PatchCoreOfficial._embed, plus optional position
        features + multi-scale averaging."""
        base = super()._embed(images)                    # (B, C, H, W)
        if 'multi_scale' in self.extensions:
            # second pass at half resolution; bilinear-upsample then average
            half = F.interpolate(images, scale_factor=0.5, mode='bilinear',
                                 align_corners=False)
            half_emb = super()._embed(half)
            half_up = F.interpolate(half_emb, size=base.shape[-2:],
                                    mode='bilinear', align_corners=False)
            base = 0.5 * (base + half_up)
        if 'position_features' in self.extensions:
            B, C, H, W = base.shape
            if self._pos_grid is None or self._pos_grid.shape[-2:] != (H, W):
                self._build_pos_grid(H, W)
            pos = self._pos_grid.expand(B, -1, -1, -1)
            base = torch.cat([base, pos], dim=1)
        return base

    @torch.no_grad()
    def predict(self, images: torch.Tensor,
                target_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        if 'softmax_reweight' not in self.extensions:
            return super().predict(images, target_size)
        # Reweighted scoring: query K nearest neighbours, then softmax-weight.
        emb = self._embed(images)
        B, C, H, W = emb.shape
        patches = (emb.permute(0, 2, 3, 1)
                   .reshape(-1, C)
                   .cpu()
                   .numpy()
                   .astype(np.float32, copy=False))
        K = max(self.reweight_k, self.k)
        dists, _ = self._faiss_index.search(patches, K)
        d = np.sqrt(np.maximum(dists, 0.0))               # (N, K)
        # weights: paper formulation w_i = exp(-d_i / d_max)
        d_max = d[:, -1:] + 1e-6
        w = np.exp(-d / d_max)
        w = w / w.sum(axis=1, keepdims=True)
        # The PatchCore paper actually multiplies the *closest* distance by
        # (1 - softmax_max). We keep it simpler: weighted mean of the top-K.
        patch_scores = (w * d).sum(axis=1)
        patch_scores_t = torch.from_numpy(
            patch_scores.reshape(B, 1, H, W)
        ).float()
        image_scores = patch_scores_t.reshape(B, -1).max(dim=1).values
        score_maps = F.interpolate(
            patch_scores_t,
            size=target_size,
            mode='bilinear',
            align_corners=False,
        )[:, 0].numpy()
        return score_maps, image_scores.numpy()

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'coreset_features': self.coreset_features,
            'feature_map_shape': self.feature_map_shape,
            'backbone': self.backbone_name,
            'layers': self.layers,
            'input_size': self.input_size,
            'extensions': self.extensions,
            'position_weight': self.position_weight,
            'reweight_k': self.reweight_k,
        }, path)

    def load(self, path: str) -> None:
        blob = torch.load(path, map_location='cpu', weights_only=False)
        self.coreset_features = blob['coreset_features']
        self.feature_map_shape = blob.get('feature_map_shape')
        self.backbone_name = blob.get('backbone', self.backbone_name)
        self.layers = tuple(blob.get('layers', self.layers))
        self.input_size = int(blob.get('input_size', self.input_size))
        self.extensions = list(blob.get('extensions', self.extensions))
        self.position_weight = float(blob.get('position_weight', self.position_weight))
        self.reweight_k = int(blob.get('reweight_k', self.reweight_k))
        self._build_index()
