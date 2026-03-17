"""t-SNE / UMAP visualization of embeddings and clusters."""

from __future__ import annotations

from pathlib import Path

import torch
import numpy as np


def visualize_embeddings(
    model,
    tokenizer=None,
    method: str = "umap",
    output_path: str = "figures/embeddings.png",
    n_tokens: int = 1000,
) -> None:
    """Visualize token embeddings colored by cluster assignment.

    Args:
        model: The HCLM-D model.
        tokenizer: Optional tokenizer for token labels.
        method: "umap" or "tsne".
        output_path: Where to save the figure.
        n_tokens: Number of tokens to visualize.
    """
    import matplotlib.pyplot as plt

    model.eval()

    with torch.no_grad():
        # Get embeddings for first n_tokens
        token_ids = torch.arange(min(n_tokens, model.config.vocab_size)).unsqueeze(0)
        e_local = model.embedding.local_embedding(token_ids)  # [1, N, D]
        embeddings = e_local.squeeze(0).cpu().numpy()

        # Get cluster assignments
        from model.embedding.composite import CompositeEmbedding
        if isinstance(model.embedding, CompositeEmbedding):
            fine_weights = model.embedding.fine_router(e_local)
            clusters = fine_weights.squeeze(0).argmax(dim=-1).cpu().numpy()
        else:
            clusters = np.zeros(len(embeddings))

    # Dimensionality reduction
    if method == "umap":
        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=42)
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)

    coords = reducer.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=clusters, cmap="tab20", alpha=0.6, s=10,
    )
    ax.set_title(f"Token Embeddings ({method.upper()}) colored by cluster")
    plt.colorbar(scatter, ax=ax, label="Cluster ID")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
