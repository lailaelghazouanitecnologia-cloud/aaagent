"""Gate activation distribution analysis."""

from __future__ import annotations

import torch


@torch.no_grad()
def analyze_gate(
    model,
    dataloader,
    device: str = "cuda",
    max_batches: int = 50,
) -> dict[str, float]:
    """Analyze gate activation distribution across the dataset.

    Args:
        model: The HCLM-D model.
        dataloader: Data loader.
        device: Device.
        max_batches: Limit number of batches to process.

    Returns:
        Dict with gate statistics.
    """
    from model.embedding.composite import CompositeEmbedding

    model.eval()

    if not isinstance(model.embedding, CompositeEmbedding) or model.embedding.gate is None:
        return {"error": "Model has no gate (flat embedding or gate disabled)"}

    all_gate_values = []

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        e_local = model.embedding.local_embedding(input_ids)
        gate_vals = model.embedding.gate(e_local)
        all_gate_values.append(gate_vals.cpu())

    if not all_gate_values:
        return {"error": "No data processed"}

    gate_tensor = torch.cat(all_gate_values, dim=0)

    return {
        "mean": gate_tensor.mean().item(),
        "std": gate_tensor.std().item(),
        "min": gate_tensor.min().item(),
        "max": gate_tensor.max().item(),
        "median": gate_tensor.median().item(),
        "pct_at_floor": (gate_tensor <= 0.11).float().mean().item(),
        "pct_above_half": (gate_tensor > 0.5).float().mean().item(),
    }
