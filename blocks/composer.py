"""Block Composer — creates new blocks by composing existing ones.

compose(A, B, C) is NOT a simple average. It produces a structured
embedding that captures:
- Semantic blend (position-weighted)
- IO profile (input from first, output from last)
- Composition metadata via a small learned network
"""

from __future__ import annotations

import torch
import torch.nn as nn

from blocks.block import Block, BlockType


class CompositionNetwork(nn.Module):
    """Small network that learns to compose meta-embeddings.

    Takes concatenated meta-embeddings of component blocks and
    produces a composed meta-embedding. ~50K params.
    """

    def __init__(self, meta_dim: int = 128, max_components: int = 8):
        super().__init__()
        self.meta_dim = meta_dim
        self.max_components = max_components

        # Input: meta_dim × max_components (padded) → meta_dim
        self.net = nn.Sequential(
            nn.Linear(meta_dim * max_components, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, meta_dim),
        )

    def forward(self, meta_embeddings: list[torch.Tensor]) -> torch.Tensor:
        """Compose meta-embeddings from component blocks.

        Args:
            meta_embeddings: List of [meta_dim] tensors.

        Returns:
            Composed meta-embedding, [meta_dim].
        """
        device = meta_embeddings[0].device

        # Pad to max_components
        padded = torch.zeros(self.max_components, self.meta_dim, device=device)
        n = min(len(meta_embeddings), self.max_components)
        for i in range(n):
            padded[i] = meta_embeddings[i]

        # Flatten and forward
        flat = padded.reshape(-1)  # [max_components × meta_dim]
        return self.net(flat)


class BlockComposer:
    """Composes blocks into new compound blocks.

    compose(A, B, C) produces a new block with:
    - embedding: position-weighted blend of component embeddings
    - meta: composed via CompositionNetwork
    - slots: empty (filled during instantiation)
    """

    def __init__(
        self,
        embed_dim: int = 384,
        meta_dim: int = 128,
        max_components: int = 8,
    ):
        self.embed_dim = embed_dim
        self.meta_dim = meta_dim
        self.composition_net = CompositionNetwork(meta_dim, max_components)

    def compose(
        self,
        blocks: list[Block],
        name: str = "",
        n_open_slots: int = 0,
    ) -> Block:
        """Create a new block from a sequence of component blocks.

        Args:
            blocks: Ordered list of component blocks.
            name: Optional name for the composed block.
            n_open_slots: Number of empty slots to add (for templating).

        Returns:
            New Block with composed embedding and meta-embedding.
        """
        if not blocks:
            raise ValueError("Cannot compose empty list of blocks")

        # Compose embedding: position-weighted average with decay
        weights = self._position_weights(len(blocks))
        device = blocks[0].embedding.device

        composed_emb = torch.zeros(self.embed_dim, device=device)
        for w, block in zip(weights, blocks):
            composed_emb += w * block.embedding

        # Compose meta-embedding via learned network
        meta_embs = []
        for block in blocks:
            if block.meta_embedding is not None:
                meta_embs.append(block.meta_embedding)
            else:
                meta_embs.append(torch.zeros(self.meta_dim, device=device))

        composed_meta = self.composition_net(meta_embs)

        # Build slots: component blocks as filled slots + open slots
        slots: list[Block | None] = list(blocks) + [None] * n_open_slots

        max_depth = max(b.depth for b in blocks) + 1

        return Block(
            block_type=BlockType.TEMPLATE if n_open_slots > 0 else BlockType.PLAN,
            embedding=composed_emb,
            slots=slots,
            meta_embedding=composed_meta,
            name=name,
            depth=max_depth,
        )

    def _position_weights(self, n: int) -> list[float]:
        """Position-weighted decay for embedding composition.

        First and last positions get more weight (input/output emphasis).
        """
        if n == 1:
            return [1.0]
        if n == 2:
            return [0.6, 0.4]

        weights = []
        for i in range(n):
            # U-shaped: higher at edges, lower in middle
            edge_dist = min(i, n - 1 - i)
            w = 1.0 / (1.0 + edge_dist * 0.3)
            weights.append(w)

        # Normalize
        total = sum(weights)
        return [w / total for w in weights]
