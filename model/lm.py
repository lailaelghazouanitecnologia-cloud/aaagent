"""Full HCLM-D model: embedding → backbone → head."""

from __future__ import annotations

import torch
import torch.nn as nn

from model.config import ModelConfig
from model.embedding.composite import CompositeEmbedding, FlatEmbedding
from model.transformer.backbone import TransformerBackbone
from model.head import LMHead


class HCLMD(nn.Module):
    """HCLM-D: Hierarchical Clustered Language Model with Masked Diffusion.

    Combines hierarchical clustered embeddings with a bidirectional Transformer
    trained via masked diffusion (LLaDA-style).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Embedding
        if config.embedding_type == "flat":
            self.embedding = FlatEmbedding(
                vocab_size=config.vocab_size,
                embed_dim=config.embed_dim,
                max_seq_len=config.max_seq_len,
                mask_token_id=config.mask_token_id,
            )
        else:
            self.embedding = CompositeEmbedding(
                vocab_size=config.vocab_size,
                embed_dim=config.embed_dim,
                max_seq_len=config.max_seq_len,
                fine_clusters=config.fine_clusters,
                coarse_clusters=config.coarse_clusters,
                gate_min=config.gate_min,
                alpha_init=config.alpha_init,
                beta_init=config.beta_init,
                use_gate=config.use_gate,
                use_hierarchy=config.use_hierarchy,
                mask_token_id=config.mask_token_id,
            )

        # Transformer backbone
        self.backbone = TransformerBackbone(
            n_layers=config.n_layers,
            embed_dim=config.embed_dim,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            activation=config.activation,
            norm=config.norm,
            causal=config.causal,
        )

        # LM head
        self.head = LMHead(
            embed_dim=config.embed_dim,
            vocab_size=config.vocab_size,
            tie_weights=config.tie_weights,
        )

        # Weight tying
        if config.tie_weights:
            self.head.set_weight(self.embedding.weight)

    def forward(
        self,
        masked_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        gate_override: float | None = None,
        router_temperature: float = 1.0,
    ) -> torch.Tensor:
        """Forward pass: embed → transform → project to logits.

        Args:
            masked_ids: Input token IDs (with masking applied), shape [B, S].
            attention_mask: Padding mask, shape [B, S]. 1 = real, 0 = pad.
            gate_override: Override gate value (for structural warmup).
            router_temperature: Softmax temperature for routers (lower = sharper).

        Returns:
            Logits over vocabulary, shape [B, S, V].
        """
        # Embed (pass gate_override + temperature only if hierarchical)
        if isinstance(self.embedding, CompositeEmbedding):
            h = self.embedding(
                masked_ids,
                gate_override=gate_override,
                router_temperature=router_temperature,
            )
        else:
            h = self.embedding(masked_ids)

        # Transform
        h = self.backbone(h, attention_mask)

        # Project to vocab
        logits = self.head(h)

        return logits

    def count_parameters(self) -> dict[str, int]:
        """Count parameters by component."""
        counts = {}

        emb_params = sum(p.numel() for p in self.embedding.parameters())
        counts["embedding"] = emb_params

        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        counts["backbone"] = backbone_params

        if not self.config.tie_weights:
            head_params = sum(p.numel() for p in self.head.parameters())
            counts["head"] = head_params
        else:
            counts["head"] = 0  # Tied with embedding

        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts
