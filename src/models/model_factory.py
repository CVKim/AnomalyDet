"""Build an anomaly-detection model from a config dict.

Selection priority (CLI override > YAML > default):
    model_name = 'patchcore_official' | 'patchcore_plus' | 'padim'

All three implementations expose the same public API:
    fit(dataloader)          -> trains the model from a DataLoader yielding
                                {'image': tensor} batches
    predict(images, target_size)  -> (score_maps_low: np.ndarray (B, H, W),
                                      image_scores:    np.ndarray (B,))
    save(path) / load(path)
so the rest of the runner is model-agnostic.
"""
from __future__ import annotations

from typing import Optional


def build_model(cfg: dict, device: str = 'cuda',
                model_name_override: Optional[str] = None,
                num_nn_override: Optional[int] = None,
                fp16: bool = False):
    """Returns a fitted-on-demand anomaly model instance.

    cfg keys read:
      model_name           one of 'patchcore_official', 'patchcore_plus', 'padim'
                           (default 'patchcore_official' for backwards compat)
      model_extensions     list[str] for patchcore_plus
      backbone, layers, input_size                  shared
      coreset_ratio, coreset_projection_dim,
      anomaly_score_num_nn                          PatchCore family only
      position_weight, reweight_k                   patchcore_plus extras
      proj_dim, cov_eps                             padim extras
    """
    name = (model_name_override or cfg.get('model_name')
            or 'patchcore_official').lower()

    backbone = cfg['backbone']
    layers = tuple(cfg['layers'])
    input_size = int(cfg['input_size'])

    if name in ('patchcore_official', 'patchcore'):
        from src.models.patchcore_official import PatchCoreOfficial
        k = num_nn_override or int(cfg.get('anomaly_score_num_nn', 1))
        return PatchCoreOfficial(
            backbone=backbone, layers=layers, input_size=input_size,
            coreset_ratio=float(cfg.get('coreset_ratio', 0.1)),
            coreset_projection_dim=int(cfg.get('coreset_projection_dim', 128)),
            anomaly_score_num_nn=k,
            device=device, fp16=fp16,
        )

    if name == 'patchcore_plus':
        from src.models.patchcore_plus import PatchCorePlus
        k = num_nn_override or int(cfg.get('anomaly_score_num_nn', 1))
        return PatchCorePlus(
            backbone=backbone, layers=layers, input_size=input_size,
            coreset_ratio=float(cfg.get('coreset_ratio', 0.1)),
            coreset_projection_dim=int(cfg.get('coreset_projection_dim', 128)),
            anomaly_score_num_nn=k,
            extensions=list(cfg.get('model_extensions', [])),
            position_weight=float(cfg.get('position_weight', 0.1)),
            reweight_k=int(cfg.get('reweight_k', 9)),
            device=device, fp16=fp16,
        )

    if name == 'padim':
        from src.models.padim import PaDiM
        return PaDiM(
            backbone=backbone, layers=layers, input_size=input_size,
            proj_dim=int(cfg.get('proj_dim', 100)),
            cov_eps=float(cfg.get('cov_eps', 0.01)),
            device=device,
        )

    raise ValueError(f'Unknown model_name: {name!r}. '
                     'Choose one of patchcore_official, patchcore_plus, padim.')
