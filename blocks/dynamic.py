"""Dynamic block creation — runtime block/embedding generation.

The system can create new blocks dynamically:
1. From pattern detection (recurring sequences → templates)
2. From model output (CREATE tokens → new blocks)
3. From VM execution (compose opcodes → new blocks)

Key constraint: dynamic blocks don't modify model weights.
They create new entries in the embedding/block registry.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from blocks.block import Block, BlockType
from blocks.registry import BlockRegistry
from blocks.composer import BlockComposer
from blocks.meta_embedding import MetaEmbeddingGenerator

logger = logging.getLogger(__name__)


class HyperNetDelta(nn.Module):
    """Small hypernetwork that generates embedding corrections.

    ΔE_dyn = HyperNet(z_ctx, z_plan, z_exec)
    E_final = E_total + gate_dyn ⊙ ΔE_dyn

    Produces a per-token delta conditioned on:
    - Current context embedding
    - Plan state (if in a plan)
    - Execution feedback (if VM active)
    - Hierarchical depth
    """

    def __init__(self, embed_dim: int = 384, context_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim

        # Input: embed_dim (context) + context_dim (plan/exec state)
        self.net = nn.Sequential(
            nn.Linear(embed_dim + context_dim, 256),
            nn.SiLU(),
            nn.Linear(256, embed_dim),
            nn.Tanh(),  # Bounded delta
        )

        # Learnable gate for the dynamic correction
        self.gate = nn.Sequential(
            nn.Linear(embed_dim + context_dim, 64),
            nn.SiLU(),
            nn.Linear(64, embed_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        embedding: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Compute dynamic embedding correction.

        Args:
            embedding: [batch, seq, embed_dim] base embedding
            context: [batch, seq, context_dim] contextual state

        Returns:
            Corrected embedding [batch, seq, embed_dim]
        """
        combined = torch.cat([embedding, context], dim=-1)
        delta = self.net(combined)
        gate = self.gate(combined)
        return embedding + gate * delta


class DynamicBlockManager:
    """Manages runtime creation and lifecycle of dynamic blocks.

    Coordinates between the pattern detector, composer, meta-embedding
    generator, and block registry to create new blocks at runtime.
    """

    def __init__(
        self,
        registry: BlockRegistry,
        composer: BlockComposer,
        meta_gen: MetaEmbeddingGenerator,
        embed_dim: int = 384,
        meta_dim: int = 128,
        max_dynamic_blocks: int = 1000,
    ):
        self.registry = registry
        self.composer = composer
        self.meta_gen = meta_gen
        self.embed_dim = embed_dim
        self.meta_dim = meta_dim
        self.max_dynamic_blocks = max_dynamic_blocks

        self.hyper_net = HyperNetDelta(embed_dim, context_dim=meta_dim)
        self._dynamic_count = 0

    def create_from_sequence(
        self,
        blocks: list[Block],
        name: str = "",
        n_open_slots: int = 0,
    ) -> Block:
        """Create a new block by composing a sequence of existing blocks.

        1. Compose embeddings via BlockComposer
        2. Generate meta-embedding via MetaEmbeddingGenerator
        3. Register in BlockRegistry
        """
        composed = self.composer.compose(blocks, name=name, n_open_slots=n_open_slots)

        # Generate meta-embedding
        meta = self.meta_gen.generate_for_block(composed)
        composed.meta_embedding = meta

        # Register
        self.registry.register(composed)
        self._dynamic_count += 1

        return composed

    def create_token_block(
        self,
        token_id: int,
        embedding: torch.Tensor,
    ) -> Block:
        """Create a leaf block for a single token."""
        block = Block(
            block_type=BlockType.TOKEN,
            embedding=embedding,
            token_id=token_id,
        )
        # Generate meta
        meta = self.meta_gen.generate_for_block(block)
        block.meta_embedding = meta
        return block

    def create_template(
        self,
        name: str,
        filled_slots: list[Block],
        n_empty_slots: int,
        base_embedding: Optional[torch.Tensor] = None,
    ) -> Block:
        """Create a template block with mixed filled/empty slots.

        Args:
            name: Template name.
            filled_slots: Pre-filled slot blocks.
            n_empty_slots: Number of empty (maskable) slots.
            base_embedding: Optional pre-computed embedding.
        """
        slots: list[Block | None] = list(filled_slots) + [None] * n_empty_slots

        if base_embedding is None and filled_slots:
            # Derive embedding from filled slots
            base_embedding = torch.stack([b.embedding for b in filled_slots]).mean(0)
        elif base_embedding is None:
            base_embedding = torch.zeros(self.embed_dim)

        template = Block(
            block_type=BlockType.TEMPLATE,
            embedding=base_embedding,
            slots=slots,
            name=name,
            depth=max((b.depth for b in filled_slots), default=0) + 1,
        )

        meta = self.meta_gen.generate_for_block(template)
        template.meta_embedding = meta

        self.registry.register(template)
        self._dynamic_count += 1

        return template

    @property
    def n_dynamic(self) -> int:
        return self._dynamic_count
