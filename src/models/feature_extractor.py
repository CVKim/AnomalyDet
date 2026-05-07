from typing import Dict, Sequence

import torch
import torch.nn as nn
import torchvision.models as tvm


_BACKBONES = {
    'wide_resnet50_2': (tvm.wide_resnet50_2, getattr(tvm, 'Wide_ResNet50_2_Weights', None)),
    'resnet50': (tvm.resnet50, getattr(tvm, 'ResNet50_Weights', None)),
    'resnet18': (tvm.resnet18, getattr(tvm, 'ResNet18_Weights', None)),
}


class FeatureExtractor(nn.Module):
    """Frozen ImageNet backbone exposing intermediate feature maps via forward hooks."""

    def __init__(self, backbone: str = 'wide_resnet50_2',
                 layers: Sequence[str] = ('layer2', 'layer3')):
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f'Unsupported backbone: {backbone}. '
                             f'Choose from {sorted(_BACKBONES)}')
        ctor, weights_enum = _BACKBONES[backbone]
        weights = weights_enum.DEFAULT if weights_enum is not None else None
        self.model = ctor(weights=weights)
        self.layers = tuple(layers)

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
