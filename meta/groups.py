"""Parameter group definitions for meta-optimizer."""

from __future__ import annotations

import torch.nn as nn


def get_parameter_groups(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    """Define parameter groups for the meta-optimizer.

    Groups:
        - embedding: E_local weights
        - router: Fine and coarse router weights
        - centroids: Fine and coarse centroid matrices
        - gate: Gate network parameters
        - backbone: Transformer parameters

    Args:
        model: The HCLM-D model.

    Returns:
        Dict mapping group names to lists of parameters.
    """
    groups: dict[str, list[nn.Parameter]] = {
        "embedding": [],
        "router": [],
        "centroids": [],
        "gate": [],
        "backbone": [],
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "local_embedding" in name:
            groups["embedding"].append(param)
        elif "router" in name:
            groups["router"].append(param)
        elif "centroids" in name or "alpha" in name or "beta" in name:
            groups["centroids"].append(param)
        elif "gate" in name:
            groups["gate"].append(param)
        else:
            groups["backbone"].append(param)

    return groups
