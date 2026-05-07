from typing import Dict, Sequence

import torch
import torch.nn as nn
import torchvision.models as tvm


_RESNET_BACKBONES = {
    'wide_resnet50_2': (tvm.wide_resnet50_2, getattr(tvm, 'Wide_ResNet50_2_Weights', None)),
    'resnet50': (tvm.resnet50, getattr(tvm, 'ResNet50_Weights', None)),
    'resnet18': (tvm.resnet18, getattr(tvm, 'ResNet18_Weights', None)),
}

_DINOV2_BACKBONES = {
    'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14',
}


class ResNetFeatureExtractor(nn.Module):
    """Frozen ResNet-family backbone exposing intermediate feature maps
    via forward hooks on named modules."""

    def __init__(self, backbone: str = 'wide_resnet50_2',
                 layers: Sequence[str] = ('layer2', 'layer3')):
        super().__init__()
        if backbone not in _RESNET_BACKBONES:
            raise ValueError(f'Unsupported ResNet backbone: {backbone}.')
        ctor, weights_enum = _RESNET_BACKBONES[backbone]
        weights = weights_enum.DEFAULT if weights_enum is not None else None
        self.model = ctor(weights=weights)
        self.layers = tuple(str(l) for l in layers)

        self._features: Dict[str, torch.Tensor] = {}
        self._handles = []
        for name, module in self.model.named_modules():
            if name in self.layers:
                self._handles.append(module.register_forward_hook(self._hook(name)))

        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def _hook(self, name: str):
        def fn(_module, _inp, out):
            self._features[name] = out
        return fn

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._features = {}
        _ = self.model(x)
        missing = [n for n in self.layers if n not in self._features]
        if missing:
            raise RuntimeError(f'Hooks did not capture layers: {missing}')
        return {n: self._features[n] for n in self.layers}


class DinoV2FeatureExtractor(nn.Module):
    """Frozen DINOv2 ViT backbone, returning intermediate transformer
    block outputs reshaped back to (B, D, H, W) so the PatchCore pipeline
    can treat them like CNN feature maps. Layers are block indices, e.g.
    (5, 11) for ViT-S/14 which has 12 blocks (0-indexed).

    Loaded via torch.hub from facebookresearch/dinov2 the first time and
    cached under ~/.cache/torch/hub/.
    """

    def __init__(self, backbone: str = 'dinov2_vits14',
                 layers: Sequence = (5, 11)):
        super().__init__()
        if backbone not in _DINOV2_BACKBONES:
            raise ValueError(f'Unsupported DINOv2 backbone: {backbone}.')
        self.model = torch.hub.load('facebookresearch/dinov2', backbone, verbose=False)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        self.layers = tuple(int(l) for l in layers)
        self.patch_size = int(getattr(self.model, 'patch_size', 14))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        H, W = x.shape[-2:]
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f'DINOv2 input must be divisible by patch_size={self.patch_size}; '
                f'got {H}x{W}.'
            )
        # n=list of indices, reshape=True returns (B, D, h, w) per layer
        outs = self.model.get_intermediate_layers(
            x, n=list(self.layers), reshape=True
        )
        return {f'block{idx}': outs[i] for i, idx in enumerate(self.layers)}


def build_feature_extractor(backbone: str, layers: Sequence) -> nn.Module:
    if backbone in _RESNET_BACKBONES:
        return ResNetFeatureExtractor(backbone=backbone, layers=layers)
    if backbone in _DINOV2_BACKBONES:
        return DinoV2FeatureExtractor(backbone=backbone, layers=layers)
    raise ValueError(
        f'Unknown backbone {backbone!r}. '
        f'Choose ResNet {sorted(_RESNET_BACKBONES)} '
        f'or DINOv2 {sorted(_DINOV2_BACKBONES)}.'
    )


# Back-compat: previously the only class was FeatureExtractor (ResNet-only).
FeatureExtractor = ResNetFeatureExtractor
