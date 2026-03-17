"""NLL / bits-per-byte evaluation on validation set."""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader

from data.masking import DiffusionMasker
from losses.diffusion_loss import diffusion_loss


@torch.no_grad()
def compute_nll(
    model,
    dataloader: DataLoader,
    mask_token_id: int = 0,
    n_samples: int = 10,
    device: str = "cuda",
) -> dict[str, float]:
    """Compute approximate NLL via multiple masking samples.

    For masked diffusion models, NLL is estimated by averaging the diffusion
    loss over multiple random masking ratios per example.

    Args:
        model: The HCLM-D model.
        dataloader: Validation data loader.
        mask_token_id: [MASK] token ID.
        n_samples: Number of masking ratio samples per example.
        device: Device to run on.

    Returns:
        Dict with 'nll', 'ppl' (perplexity), 'bpb' (bits per byte).
    """
    model.eval()
    masker = DiffusionMasker(mask_token_id=mask_token_id)
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        B, S = input_ids.shape

        batch_loss = 0.0
        for _ in range(n_samples):
            masked_ids, mask = masker.mask_batch(input_ids, attention_mask)
            logits = model(masked_ids, attention_mask)
            loss = diffusion_loss(logits, input_ids, mask)
            batch_loss += loss.item()

        total_loss += batch_loss / n_samples * B * S
        total_tokens += (attention_mask.sum()).item()

    avg_nll = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_nll, 100))  # Cap to avoid overflow
    bpb = avg_nll / math.log(2)

    return {"nll": avg_nll, "ppl": ppl, "bpb": bpb}
