"""PatchCore: memory bank of locally-aggregated mid-block features
from a frozen ImageNet backbone, with coreset subsampling.

Reference: Roth et al., "Towards Total Recall in Industrial Anomaly
Detection" (CVPR 2022).
"""
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.models.feature_extractor import build_feature_extractor
from src.utils.coreset import k_center_greedy


class PatchCore:
    def __init__(self,
                 backbone: str = 'wide_resnet50_2',
                 layers: Sequence[str] = ('layer2', 'layer3'),
                 input_size: int = 224,
                 coreset_ratio: float = 0.1,
                 coreset_projection_dim: int = 128,
                 reweight_k: int = 0,
                 smooth_kernel: int = 11,
                 smooth_sigma: float = 4.0,
                 device: str = 'cuda'):
        self.input_size = input_size
        self.coreset_ratio = coreset_ratio
        self.coreset_projection_dim = coreset_projection_dim
        # K for the PatchCore softmax score reweighting; 0 disables.
        self.reweight_k = int(reweight_k)
        self.smooth_kernel = int(smooth_kernel)
        self.smooth_sigma = float(smooth_sigma)
        self.device = device

        self.extractor = build_feature_extractor(backbone=backbone, layers=layers).to(device).eval()
        self.layers = tuple(layers)

        self.memory_bank: Optional[torch.Tensor] = None
        self.feature_map_size: Optional[Tuple[int, int]] = None
        self.embed_dim: Optional[int] = None
        # Calibration statistics computed on the training set after fit().
        # train_pixel_max is the recall-friendly default threshold;
        # train_pixel_p99 is a stricter (precision-friendly) threshold.
        self.train_pixel_max: Optional[float] = None
        self.train_pixel_p99: Optional[float] = None
        self.train_pixel_p999: Optional[float] = None
        self.train_image_max: Optional[float] = None

    @torch.no_grad()
    def _embed(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.extractor(images)
        # Both backends keep insertion order matching self.layers, so
        # iterating values() preserves the configured layer ordering
        # without us having to know whether keys are 'layer2' or 'block5'.
        ordered = list(feats.values())
        target_h, target_w = ordered[0].shape[-2:]
        aligned = []
        for f in ordered:
            if f.shape[-2:] != (target_h, target_w):
                f = F.interpolate(f, size=(target_h, target_w),
                                  mode='bilinear', align_corners=False)
            f = F.avg_pool2d(f, kernel_size=3, stride=1, padding=1)
            aligned.append(f)
        return torch.cat(aligned, dim=1)

    @torch.no_grad()
    def fit(self, dataloader) -> None:
        all_patches = []
        for batch in tqdm(dataloader, desc='extracting train features'):
            images = batch['image'].to(self.device, non_blocking=True)
            emb = self._embed(images)
            B, C, H, W = emb.shape
            self.feature_map_size = (H, W)
            self.embed_dim = C
            patches = emb.permute(0, 2, 3, 1).reshape(-1, C).contiguous()
            all_patches.append(patches.cpu())

        all_patches = torch.cat(all_patches, dim=0)
        n_total = all_patches.shape[0]
        n_select = max(1, int(n_total * self.coreset_ratio))
        print(f'patches: {n_total} -> coreset target: {n_select}')

        indices = k_center_greedy(
            all_patches,
            n_select=n_select,
            projection_dim=self.coreset_projection_dim,
            device=self.device,
        )
        self.memory_bank = all_patches[indices].to(self.device).contiguous()
        print(f'memory bank: {tuple(self.memory_bank.shape)}')

    @torch.no_grad()
    def calibrate(self, dataloader) -> None:
        """Run predict over the (normal-only) training set to record the
        pixel-score distribution. The max becomes the recall-first
        threshold default at inference time.
        """
        all_pix = []
        all_img = []
        for batch in tqdm(dataloader, desc='calibrating on train'):
            images = batch['image'].to(self.device, non_blocking=True)
            heatmaps, image_scores = self.predict(images)
            all_pix.append(heatmaps.flatten().cpu().numpy())
            all_img.append(image_scores.cpu().numpy())
        all_pix = np.concatenate(all_pix)
        all_img = np.concatenate(all_img)
        self.train_pixel_max = float(all_pix.max())
        self.train_pixel_p99 = float(np.percentile(all_pix, 99.0))
        self.train_pixel_p999 = float(np.percentile(all_pix, 99.9))
        self.train_image_max = float(all_img.max())
        print(f'calibration: pixel max={self.train_pixel_max:.4f} '
              f'p99.9={self.train_pixel_p999:.4f} '
              f'p99={self.train_pixel_p99:.4f}, '
              f'image max={self.train_image_max:.4f}')

    @torch.no_grad()
    def predict(self, images: torch.Tensor,
                chunk_size: int = 4096) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.memory_bank is None:
            raise RuntimeError('Memory bank is not built. Call fit() or load() first.')
        emb = self._embed(images)
        B, C, H, W = emb.shape
        patches = emb.permute(0, 2, 3, 1).reshape(B * H * W, C)

        scores = torch.empty(patches.shape[0], device=self.device)
        K = max(1, self.reweight_k)
        K = min(K, self.memory_bank.shape[0])
        for start in range(0, patches.shape[0], chunk_size):
            chunk = patches[start:start + chunk_size]
            dist = torch.cdist(chunk.unsqueeze(0), self.memory_bank.unsqueeze(0)).squeeze(0)
            if self.reweight_k > 1:
                # PatchCore K-NN softmax reweighting: a patch sitting far
                # from a sparse region of the memory bank gets a bigger
                # boost than one near a dense cluster, sharpening the
                # anomaly response.
                topk = dist.topk(k=K, dim=1, largest=False).values  # (n, K)
                d_star = topk[:, 0]
                w = 1.0 - torch.softmax(-topk, dim=1)[:, 0]
                scores[start:start + chunk.shape[0]] = w * d_star
            else:
                scores[start:start + chunk.shape[0]] = dist.min(dim=1).values

        anomaly_map = scores.reshape(B, H, W)
        anomaly_map = F.interpolate(
            anomaly_map.unsqueeze(1),
            size=(self.input_size, self.input_size),
            mode='bilinear',
            align_corners=False,
        ).squeeze(1)
        # Lightweight smoothing approximating PatchCore's Gaussian blur step.
        if self.smooth_kernel > 1 and self.smooth_sigma > 0:
            kernel = self._gaussian_kernel(self.smooth_kernel, self.smooth_sigma).to(self.device)
            anomaly_map = F.conv2d(
                anomaly_map.unsqueeze(1),
                kernel,
                padding=kernel.shape[-1] // 2,
            ).squeeze(1)

        image_scores = anomaly_map.amax(dim=(1, 2))
        return anomaly_map, image_scores

    @staticmethod
    def _gaussian_kernel(ksize: int, sigma: float) -> torch.Tensor:
        ax = torch.arange(ksize) - (ksize - 1) / 2.0
        g = torch.exp(-(ax ** 2) / (2.0 * sigma ** 2))
        g = g / g.sum()
        kernel = torch.outer(g, g)
        return kernel.view(1, 1, ksize, ksize)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'memory_bank': self.memory_bank.detach().cpu(),
            'feature_map_size': self.feature_map_size,
            'embed_dim': self.embed_dim,
            'input_size': self.input_size,
            'layers': self.layers,
            'train_pixel_max': self.train_pixel_max,
            'train_pixel_p99': self.train_pixel_p99,
            'train_pixel_p999': self.train_pixel_p999,
            'train_image_max': self.train_image_max,
        }, path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.memory_bank = state['memory_bank'].to(self.device).contiguous()
        self.feature_map_size = state['feature_map_size']
        self.embed_dim = state['embed_dim']
        self.input_size = state['input_size']
        self.train_pixel_max = state.get('train_pixel_max')
        self.train_pixel_p99 = state.get('train_pixel_p99')
        self.train_pixel_p999 = state.get('train_pixel_p999')
        self.train_image_max = state.get('train_image_max')
