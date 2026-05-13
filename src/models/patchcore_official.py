"""Vanilla PatchCore (Roth et al., CVPR 2022) implementation that mirrors
the senior engineer's IncrementalCoresetModel reference structure:

  feature_embedder -> 3x3 local aggregation -> coreset (k-center greedy)
                   -> FAISS-backed nearest-neighbour scoring (K=1 default)
                   -> bilinear upsample to input size (NO Gaussian)

Deliberately omits all the deviations the wider AnomalyDet stack had
introduced (Gaussian smoothing, softmax score reweighting, adaptive
image-level gating, severity / pixel floor, foreground masking,
fragment-merging dilate-erode). Postprocessing is exactly thresholding
plus an optional small connected-component filter -- nothing else.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.models.feature_extractor import build_feature_extractor
from src.utils.coreset import k_center_greedy


class PatchCoreOfficial:
    def __init__(self,
                 backbone: str = 'wide_resnet50_2',
                 layers: Sequence = ('layer2', 'layer3'),
                 input_size: int = 224,
                 coreset_ratio: float = 0.1,
                 coreset_projection_dim: int = 128,
                 anomaly_score_num_nn: int = 1,
                 device: str = 'cuda',
                 fp16: bool = False):
        self.backbone_name = backbone
        self.layers = tuple(layers)
        self.input_size = int(input_size)
        self.coreset_ratio = float(coreset_ratio)
        self.coreset_projection_dim = int(coreset_projection_dim)
        self.k = int(anomaly_score_num_nn)
        self.device = device
        self.fp16 = bool(fp16)

        self.feature_embedder = build_feature_extractor(
            backbone=backbone, layers=layers,
        ).to(device).eval()
        if self.fp16 and 'cuda' in str(self.device):
            # Cast backbone to half. Inputs cast inside _embed.
            self.feature_embedder = self.feature_embedder.half()

        # State filled in by fit() / load().
        self.coreset_features: Optional[torch.Tensor] = None   # (M, D) cpu
        self.feature_map_shape: Optional[Tuple[int, int, int]] = None  # (D, H, W)
        self._faiss_index: Optional[faiss.Index] = None

    # ---- forward ---------------------------------------------------------

    @torch.no_grad()
    def _embed(self, images: torch.Tensor) -> torch.Tensor:
        """Backbone features at the configured layers, aligned to the
        first layer's spatial resolution, with 3x3 avg-pool aggregation
        (PatchCore's local neighbourhood feature).

        When `self.fp16` is set and we're on CUDA, the input + backbone
        run in half precision (~1.7x faster + half VRAM); features are
        cast back to float32 before patch aggregation so coreset /
        FAISS downstream stays numerically stable.
        """
        if self.fp16 and 'cuda' in str(self.device):
            images = images.half()
        feats = self.feature_embedder(images)
        ordered = list(feats.values())
        target_h, target_w = ordered[0].shape[-2:]
        aligned = []
        for f in ordered:
            if f.dtype != torch.float32:
                f = f.float()
            if f.shape[-2:] != (target_h, target_w):
                f = F.interpolate(f, size=(target_h, target_w),
                                  mode='bilinear', align_corners=False)
            f = F.avg_pool2d(f, kernel_size=3, stride=1, padding=1)
            aligned.append(f)
        return torch.cat(aligned, dim=1)

    # ---- fit / coreset / index ------------------------------------------

    @torch.no_grad()
    def fit(self, dataloader) -> None:
        all_patches = []
        for batch in tqdm(dataloader, desc='computing support features'):
            images = batch['image'].to(self.device, non_blocking=True)
            emb = self._embed(images)
            B, C, H, W = emb.shape
            self.feature_map_shape = (C, H, W)
            patches = emb.permute(0, 2, 3, 1).reshape(-1, C).contiguous()
            all_patches.append(patches.cpu())
        all_patches = torch.cat(all_patches, dim=0)
        n_total = all_patches.shape[0]
        n_select = max(1, int(n_total * self.coreset_ratio))
        print(f'patches: {n_total} -> coreset target: {n_select}')

        idx = k_center_greedy(
            all_patches,
            n_select=n_select,
            projection_dim=self.coreset_projection_dim,
            device=self.device,
        )
        self.coreset_features = all_patches[idx].contiguous()
        print(f'coreset features: {tuple(self.coreset_features.shape)}')
        self._build_index()

    def _build_index(self) -> None:
        if self.coreset_features is None:
            raise RuntimeError('No coreset features to index.')
        feats = self.coreset_features.numpy().astype(np.float32, copy=False)
        d = feats.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(feats)
        self._faiss_index = index

    # ---- predict ---------------------------------------------------------

    @torch.no_grad()
    def predict(self, images: torch.Tensor,
                target_size: Optional[Tuple[int, int]] = None
                ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (score_maps, image_scores).

        score_maps : (B, target_h, target_w) np.float32
        image_scores : (B,) np.float32  -- max patch L2 distance per image
        """
        if self._faiss_index is None:
            raise RuntimeError('fit() or load() must be called first.')
        if target_size is None:
            target_size = (int(images.shape[-2]), int(images.shape[-1]))

        emb = self._embed(images)
        B, C, H, W = emb.shape
        patches = (emb.permute(0, 2, 3, 1)
                   .reshape(-1, C)
                   .cpu()
                   .numpy()
                   .astype(np.float32, copy=False))
        # FAISS returns squared L2; take sqrt to match torch.cdist semantics.
        dists, _ = self._faiss_index.search(patches, self.k)
        if self.k == 1:
            patch_scores = np.sqrt(dists[:, 0])
        else:
            patch_scores = np.sqrt(dists).mean(axis=1)

        patch_scores_t = torch.from_numpy(patch_scores.reshape(B, 1, H, W)).float()
        image_scores = patch_scores_t.reshape(B, -1).max(dim=1).values
        score_maps = F.interpolate(
            patch_scores_t,
            size=target_size,
            mode='bilinear',
            align_corners=False,
        )[:, 0].numpy()
        return score_maps, image_scores.numpy()

    # ---- io --------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'coreset_features': self.coreset_features,
            'feature_map_shape': self.feature_map_shape,
            'backbone': self.backbone_name,
            'layers': self.layers,
            'input_size': self.input_size,
            'k': self.k,
        }, path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location='cpu', weights_only=False)
        self.coreset_features = state['coreset_features'].contiguous()
        self.feature_map_shape = state['feature_map_shape']
        self._build_index()
