"""k-center greedy coreset subsampling used by PatchCore.

We project to a lower dimension first (sparse Gaussian random projection,
Johnson-Lindenstrauss) so distance updates are cheap during the greedy loop.
"""
from typing import List

import numpy as np
import torch


def _random_projection(features: torch.Tensor, dim: int) -> torch.Tensor:
    n, d = features.shape
    if dim is None or dim >= d:
        return features
    g = torch.Generator(device='cpu').manual_seed(0)
    proj = torch.randn(d, dim, generator=g) / np.sqrt(dim)
    return features @ proj


@torch.no_grad()
def k_center_greedy(features: torch.Tensor, n_select: int,
                    projection_dim: int = 128,
                    device: str = 'cuda',
                    seed: int = 0) -> List[int]:
    """Greedy coreset selection.

    features: (N, D) tensor on CPU.
    Returns: list of N_select indices into `features`.
    """
    if n_select >= features.shape[0]:
        return list(range(features.shape[0]))

    proj = _random_projection(features, projection_dim).to(device)

    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, proj.shape[0]))
    selected = [start]

    min_dist = torch.cdist(proj, proj[start:start + 1]).squeeze(1)

    for _ in range(n_select - 1):
        idx = int(torch.argmax(min_dist).item())
        selected.append(idx)
        new_dist = torch.cdist(proj, proj[idx:idx + 1]).squeeze(1)
        min_dist = torch.minimum(min_dist, new_dist)

    return selected
