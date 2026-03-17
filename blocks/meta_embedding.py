"""Meta-embedding generation and management.

A meta-embedding describes WHAT a block IS, not what it DOES.
- The embedding says: "I sum two numbers"
- The meta-embedding says: "I am a commutative binary arithmetic op"

Meta-embedding structure (128d):
  TYPE (16d):         What kind of block
  COMPLEXITY (16d):   How costly to execute
  RELIABILITY (16d):  Success/failure history
  FREQUENCY (16d):    How often used
  COMPOSABILITY (16d): How well it combines with others
  DOMAIN (16d):       Semantic domain (IO, math, string, etc.)
  IO_PROFILE (16d):   Input/output signature
  TEMPORAL (16d):     When created/updated
"""

from __future__ import annotations

import torch
import torch.nn as nn

from blocks.block import Block, BlockType


class MetaEmbeddingGenerator(nn.Module):
    """Generates meta-embeddings for blocks.

    Takes a block's embedding + structural features and produces
    a 128d meta-embedding describing the block's properties.
    """

    def __init__(self, embed_dim: int = 384, meta_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.meta_dim = meta_dim

        # Structural features: block_type (one-hot 5), n_slots, n_filled, depth
        self.struct_dim = 5 + 3  # 8 structural features

        self.encoder = nn.Sequential(
            nn.Linear(embed_dim + self.struct_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, meta_dim),
        )

        # Sections of the meta-embedding (each 16d)
        self.section_names = [
            "type", "complexity", "reliability", "frequency",
            "composability", "domain", "io_profile", "temporal",
        ]
        assert meta_dim == len(self.section_names) * 16

    def forward(self, block_embedding: torch.Tensor, struct_features: torch.Tensor) -> torch.Tensor:
        """Generate meta-embedding.

        Args:
            block_embedding: [batch, embed_dim] or [embed_dim]
            struct_features: [batch, struct_dim] or [struct_dim]

        Returns:
            Meta-embedding [batch, meta_dim] or [meta_dim]
        """
        x = torch.cat([block_embedding, struct_features], dim=-1)
        return self.encoder(x)

    def extract_struct_features(self, block: Block) -> torch.Tensor:
        """Extract structural features from a Block.

        Returns tensor of shape [struct_dim].
        """
        device = block.embedding.device

        # Block type one-hot
        type_oh = torch.zeros(5, device=device)
        type_idx = {
            BlockType.TOKEN: 0, BlockType.SPAN: 1, BlockType.TEMPLATE: 2,
            BlockType.PLAN: 3, BlockType.META: 4,
        }
        type_oh[type_idx.get(block.block_type, 0)] = 1.0

        # Structural scalars (normalized)
        n_slots = min(block.n_slots / 16.0, 1.0)
        n_filled = min(block.n_filled / 16.0, 1.0) if block.n_slots > 0 else 0.0
        depth = min(block.depth / 8.0, 1.0)

        scalars = torch.tensor([n_slots, n_filled, depth], device=device)

        return torch.cat([type_oh, scalars])

    def generate_for_block(self, block: Block) -> torch.Tensor:
        """Generate meta-embedding for a single block."""
        struct = self.extract_struct_features(block)
        return self.forward(block.embedding, struct)

    def get_section(self, meta: torch.Tensor, section: str) -> torch.Tensor:
        """Extract a 16d section from a meta-embedding.

        Args:
            meta: [meta_dim] or [batch, meta_dim]
            section: One of the section names.

        Returns:
            [16] or [batch, 16]
        """
        idx = self.section_names.index(section)
        start = idx * 16
        return meta[..., start:start + 16]

    def query_by_section(
        self,
        meta_embeddings: torch.Tensor,
        section: str,
        query: torch.Tensor,
        top_k: int = 5,
    ) -> torch.Tensor:
        """Find blocks whose meta-embedding section is most similar to query.

        Args:
            meta_embeddings: [N, meta_dim] all registered meta-embeddings
            section: Which section to compare
            query: [16] query vector
            top_k: Number of results

        Returns:
            Indices of top-k most similar blocks, [top_k]
        """
        sections = self.get_section(meta_embeddings, section)  # [N, 16]
        sims = torch.nn.functional.cosine_similarity(
            sections, query.unsqueeze(0), dim=-1
        )  # [N]
        return sims.topk(min(top_k, len(sims))).indices
