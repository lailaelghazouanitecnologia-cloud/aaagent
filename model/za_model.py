"""Za v7: Unified model — embedding → backbone (RWKV bidi or Transformer) → head.

Supports both RWKV-7 bidirectional backbone (default) and Transformer
backbone (for ablation A1). The embedding and head layers are shared.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from model.config import ModelConfig
from model.embedding.composite import CompositeEmbedding, FlatEmbedding
from model.head import LMHead


class ZaModel(nn.Module):
    """Za v7: Masked diffusion LLM with RWKV backbone.

    Architecture:
      1. Composite embedding: E_local + g⊙(αE_cluster + βE_hier) + E_pos
      2. Bidirectional backbone: RWKV fwd + bwd + merge (or Transformer)
      3. Weight-tied LM head
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

        # Backbone
        if config.backbone_type == "rwkv":
            from model.rwkv.backbone import RWKVBackbone
            self.backbone = RWKVBackbone(
                n_layers=config.rwkv_layers,
                embed_dim=config.embed_dim,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                share_weights=config.share_weights,
            )
        else:
            from model.transformer.backbone import TransformerBackbone
            self.backbone = TransformerBackbone(
                n_layers=config.transformer_layers,
                embed_dim=config.embed_dim,
                n_heads=config.transformer_heads,
                d_ff=config.transformer_d_ff,
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

        # Initialize weights RWKV-7 style
        self._init_weights()

    def _init_weights(self) -> None:
        """RWKV-7 style initialization: orthogonal for projections, scaled normal for embeddings."""
        for name, p in self.named_parameters():
            if p.dim() < 2:
                continue
            if "embedding" in name or "centroid" in name:
                # Embeddings: scaled normal
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif "W_o" in name or "merge" in name or "out_proj" in name or "w2" in name:
                # Output projections: scale down by sqrt(2 * n_layers)
                n_layers = self.config.rwkv_layers * 2 if self.config.backbone_type == "rwkv" else self.config.transformer_layers
                nn.init.orthogonal_(p)
                with torch.no_grad():
                    p.mul_(1.0 / math.sqrt(2.0 * n_layers))
            elif any(k in name for k in ("W_r", "W_k", "W_v", "W_a", "W_g", "linear", "w1", "v.")):
                # Other projections: orthogonal
                nn.init.orthogonal_(p)

    def forward(
        self,
        masked_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        gate_scale: torch.Tensor | None = None,
        router_temperature: torch.Tensor | None = None,
        coarse_temperature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass: embed → backbone → head.

        Args:
            masked_ids: Input token IDs (with masking applied), shape [B, S].
            attention_mask: Padding mask [B, S]. 1 = real, 0 = pad.
            gate_scale: Scalar tensor [0,1] for structural warmup.
            router_temperature: Fine router temperature.
            coarse_temperature: Coarse router temperature.

        Returns:
            Logits over vocabulary [B, S, V].
        """
        # Embed
        if isinstance(self.embedding, CompositeEmbedding):
            h = self.embedding(
                masked_ids,
                gate_scale=gate_scale,
                router_temperature=router_temperature,
                coarse_temperature=coarse_temperature,
            )
        else:
            h = self.embedding(masked_ids)

        # Backbone
        h = self.backbone(h, attention_mask)

        # Project to vocab
        return self.head(h)

    def count_parameters(self) -> dict[str, int]:
        """Count parameters by component."""
        counts = {}
        counts["embedding"] = sum(p.numel() for p in self.embedding.parameters())
        counts["backbone"] = sum(p.numel() for p in self.backbone.parameters())
        if not self.config.tie_weights:
            counts["head"] = sum(p.numel() for p in self.head.parameters())
        else:
            counts["head"] = 0
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts
