"""Template selection loss and hash consistency loss.

L_template = CE(T_real, T_pred)
  — Train the model to select the correct template given context.

L_hash = contrastive(E_block, H_sem)
  — Blocks with the same semantic hash should have similar embeddings.
  — Different blocks should have different embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemplateLoss(nn.Module):
    """Template hash prediction loss (cosine similarity)."""

    def __init__(self, embed_dim: int = 384):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(
        self,
        pred_hash: torch.Tensor,
        true_hash: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cosine embedding loss between predicted and true hash.

        Args:
            pred_hash: [N, D] predicted hash embeddings
            true_hash: [N, D] ground truth hash embeddings (normalized)

        Returns:
            Scalar loss tensor.
        """
        # Normalize predictions
        pred_norm = pred_hash / (pred_hash.norm(dim=-1, keepdim=True) + 1e-8)
        # 1 - cos_sim as loss
        cos_sim = (pred_norm * true_hash).sum(dim=-1)
        return (1.0 - cos_sim).mean()


def template_selection_loss(
    template_logits: torch.Tensor,
    template_targets: torch.Tensor,
) -> torch.Tensor:
    """CE loss for template selection.

    Args:
        template_logits: [batch, n_positions, n_templates] predicted template scores
        template_targets: [batch, n_positions] ground truth template indices

    Returns:
        Scalar loss tensor.
    """
    if template_logits.numel() == 0:
        return torch.tensor(0.0, device=template_logits.device)

    # Flatten batch and positions
    logits_flat = template_logits.reshape(-1, template_logits.size(-1))
    targets_flat = template_targets.reshape(-1)

    # Filter valid positions (target >= 0)
    valid = targets_flat >= 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=template_logits.device)

    return F.cross_entropy(logits_flat[valid], targets_flat[valid])


def hash_consistency_loss(
    block_embeddings: torch.Tensor,
    hash_labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Contrastive loss for hash consistency.

    Blocks with the same semantic hash should have similar embeddings.
    Different blocks should have distant embeddings.

    Uses InfoNCE-style contrastive loss.

    Args:
        block_embeddings: [N, embed_dim] embeddings of N blocks
        hash_labels: [N] integer hash group labels (same label = same hash)
        temperature: Contrastive temperature.

    Returns:
        Scalar loss tensor.
    """
    N = block_embeddings.size(0)
    if N < 2:
        return torch.tensor(0.0, device=block_embeddings.device)

    # Normalize embeddings
    emb_norm = F.normalize(block_embeddings, dim=-1)

    # Cosine similarity matrix
    sim_matrix = torch.matmul(emb_norm, emb_norm.t()) / temperature  # [N, N]

    # Positive mask: same hash label
    labels_expanded = hash_labels.unsqueeze(0)  # [1, N]
    positive_mask = (labels_expanded == labels_expanded.t()).float()  # [N, N]

    # Remove self-similarity
    eye = torch.eye(N, device=block_embeddings.device)
    positive_mask = positive_mask - eye

    # If no positive pairs, return 0
    if positive_mask.sum() == 0:
        return torch.tensor(0.0, device=block_embeddings.device)

    # InfoNCE: for each anchor, sum over positives vs all negatives
    exp_sim = torch.exp(sim_matrix - eye * 1e9)  # Mask out self
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    # Mean of log-prob over positive pairs
    loss = -(positive_mask * log_prob).sum() / positive_mask.sum()

    return loss


def compose_quality_loss(
    composed_embedding: torch.Tensor,
    target_behavior: torch.Tensor,
) -> torch.Tensor:
    """Loss for quality of composed embeddings.

    The composed embedding should predict the behavior of the
    compound operation (measured by output similarity).

    Args:
        composed_embedding: [N, embed_dim] embeddings of composed blocks
        target_behavior: [N, embed_dim] target behavior vectors

    Returns:
        Scalar loss tensor (MSE).
    """
    if composed_embedding.numel() == 0:
        return torch.tensor(0.0, device=composed_embedding.device)

    return F.mse_loss(composed_embedding, target_behavior)
