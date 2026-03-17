"""Block-aware embedding layer for Za v6.

Extends CompositeEmbedding with:
- E_block: Embedding conditioned on the current block context
- E_template: Embedding inherited from the template parent
- E_meta: Embedding from the planner/meta state
- ΔE_dyn: Dynamic correction from HyperNet

Full composition:
z₀ = E_local + g⊙(α·E_cluster + β·E_hier) + E_pos + E_block + E_template + E_meta + gate_dyn⊙ΔE_dyn
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class BlockContextEncoder(nn.Module):
    """Encodes block-level context into per-token embeddings.

    Produces E_block: a per-token embedding conditioned on which
    block the token belongs to and the block's structural properties.
    """

    def __init__(self, embed_dim: int = 384, max_depth: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_depth = max_depth

        # Depth embedding (which level in the block tree)
        self.depth_embedding = nn.Embedding(max_depth, embed_dim)

        # Block type embedding (5 types: token, span, template, plan, meta)
        self.type_embedding = nn.Embedding(5, embed_dim)

        # Projection to combine depth + type → block context
        self.proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(
        self,
        depth_ids: torch.Tensor,       # [batch, seq_len] depth at each position
        type_ids: torch.Tensor,         # [batch, seq_len] block type at each position
    ) -> torch.Tensor:
        """Compute block context embeddings.

        Returns: [batch, seq_len, embed_dim]
        """
        depth_ids = depth_ids.clamp(0, self.max_depth - 1)
        type_ids = type_ids.clamp(0, 4)

        e_depth = self.depth_embedding(depth_ids)
        e_type = self.type_embedding(type_ids)

        combined = torch.cat([e_depth, e_type], dim=-1)
        return self.proj(combined)


class TemplateContextEncoder(nn.Module):
    """Encodes template parent context into slot embeddings.

    Produces E_template: the embedding inherited from the template
    that contains this token/slot.
    """

    def __init__(self, embed_dim: int = 384):
        super().__init__()
        self.embed_dim = embed_dim

        # Slot position embedding (position within parent template)
        self.slot_pos_embedding = nn.Embedding(32, embed_dim)

        # Scale factor for template inheritance
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        parent_embedding: torch.Tensor,  # [batch, seq_len, embed_dim] parent template embedding
        slot_position: torch.Tensor,      # [batch, seq_len] slot index within parent
    ) -> torch.Tensor:
        """Compute template-inherited embeddings.

        Returns: [batch, seq_len, embed_dim]
        """
        slot_position = slot_position.clamp(0, 31)
        e_slot = self.slot_pos_embedding(slot_position)

        # Combine parent embedding with slot position
        return self.scale * (parent_embedding + e_slot)


class MetaStateEncoder(nn.Module):
    """Encodes the meta-state (planner state) into embeddings.

    z_meta is a persistent latent vector updated per iteration:
    z_meta(t+1) = f(z_meta(t), plan_tokens, vm_feedback, uncertainty, block_graph)

    This encoder projects z_meta into per-token E_meta.
    """

    def __init__(self, meta_dim: int = 128, embed_dim: int = 384):
        super().__init__()
        self.meta_dim = meta_dim
        self.embed_dim = embed_dim

        # GRU for meta-state updates
        self.gru = nn.GRUCell(meta_dim, meta_dim)

        # Project meta-state to embedding space
        self.proj = nn.Sequential(
            nn.Linear(meta_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Scale for meta contribution
        self.scale = nn.Parameter(torch.tensor(0.05))

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize meta-state. [batch, meta_dim]"""
        return torch.zeros(batch_size, self.meta_dim, device=device)

    def update_state(
        self,
        z_meta: torch.Tensor,          # [batch, meta_dim]
        feedback: torch.Tensor,         # [batch, meta_dim] from plan/VM/uncertainty
    ) -> torch.Tensor:
        """Update meta-state with new information.

        Returns: updated z_meta [batch, meta_dim]
        """
        return self.gru(feedback, z_meta)

    def forward(
        self,
        z_meta: torch.Tensor,          # [batch, meta_dim]
        seq_len: int,
    ) -> torch.Tensor:
        """Project meta-state to per-token embeddings.

        Returns: [batch, seq_len, embed_dim]
        """
        e_meta = self.proj(z_meta)                    # [batch, embed_dim]
        e_meta = e_meta.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, embed_dim]
        return self.scale * e_meta


class BlockAwareEmbedding(nn.Module):
    """Full block-aware embedding for Za v6.

    Wraps CompositeEmbedding and adds block/template/meta layers.
    Only active when block structure is provided; falls back to
    CompositeEmbedding behavior otherwise.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        meta_dim: int = 128,
        max_depth: int = 8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.meta_dim = meta_dim

        # Block-level encoders (added on top of CompositeEmbedding)
        self.block_encoder = BlockContextEncoder(embed_dim, max_depth)
        self.template_encoder = TemplateContextEncoder(embed_dim)
        self.meta_encoder = MetaStateEncoder(meta_dim, embed_dim)

        # Dynamic correction hypernetwork
        self.hyper_gate = nn.Sequential(
            nn.Linear(embed_dim + meta_dim, 64),
            nn.SiLU(),
            nn.Linear(64, embed_dim),
            nn.Sigmoid(),
        )
        self.hyper_delta = nn.Sequential(
            nn.Linear(embed_dim + meta_dim, 256),
            nn.SiLU(),
            nn.Linear(256, embed_dim),
            nn.Tanh(),
        )

        # Master gate for block-level features (starts at 0, ramps up)
        self.block_gate_scale = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        base_embedding: torch.Tensor,    # [batch, seq, embed_dim] from CompositeEmbedding
        z_meta: Optional[torch.Tensor] = None,  # [batch, meta_dim]
        depth_ids: Optional[torch.Tensor] = None,  # [batch, seq]
        type_ids: Optional[torch.Tensor] = None,   # [batch, seq]
        parent_embedding: Optional[torch.Tensor] = None,  # [batch, seq, embed_dim]
        slot_position: Optional[torch.Tensor] = None,     # [batch, seq]
    ) -> torch.Tensor:
        """Compute full block-aware embedding.

        Falls back to base_embedding if no block structure provided.
        """
        z = base_embedding
        batch, seq_len, _ = z.shape
        device = z.device

        block_scale = torch.sigmoid(self.block_gate_scale)

        # Block context (if block structure available)
        if depth_ids is not None and type_ids is not None:
            e_block = self.block_encoder(depth_ids, type_ids)
            z = z + block_scale * e_block

        # Template parent context
        if parent_embedding is not None and slot_position is not None:
            e_template = self.template_encoder(parent_embedding, slot_position)
            z = z + block_scale * e_template

        # Meta-state
        if z_meta is not None:
            e_meta = self.meta_encoder(z_meta, seq_len)
            z = z + block_scale * e_meta

            # Dynamic correction: ΔE = gate ⊙ delta
            meta_expanded = z_meta.unsqueeze(1).expand(-1, seq_len, -1)
            combined = torch.cat([z, meta_expanded], dim=-1)
            gate = self.hyper_gate(combined)
            delta = self.hyper_delta(combined)
            z = z + block_scale * gate * delta

        return z
