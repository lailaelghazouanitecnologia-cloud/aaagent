"""Model configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Full configuration for HCLM-D model."""

    # Vocabulary and dimensions
    vocab_size: int = 8192
    embed_dim: int = 384
    max_seq_len: int = 512
    mask_token_id: int = 0

    # Embedding
    embedding_type: str = "hierarchical"  # "flat" or "hierarchical"
    fine_clusters: int = 64
    coarse_clusters: int = 8
    gate_min: float = 0.1
    alpha_init: float = 0.1
    beta_init: float = 0.05
    use_gate: bool = True
    use_hierarchy: bool = True

    # Transformer
    n_layers: int = 8
    n_heads: int = 6
    d_ff: int = 1536
    dropout: float = 0.1
    activation: str = "swiglu"
    norm: str = "rmsnorm"
    causal: bool = False

    # Head
    tie_weights: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        """Create config from a nested config dictionary."""
        model_cfg = d.get("model", d)
        emb = model_cfg.get("embedding", {})
        tfm = model_cfg.get("transformer", {})
        head = model_cfg.get("head", {})

        return cls(
            vocab_size=model_cfg.get("vocab_size", 8192),
            embed_dim=model_cfg.get("embed_dim", 384),
            max_seq_len=model_cfg.get("max_seq_len", 512),
            mask_token_id=model_cfg.get("mask_token_id", 0),
            embedding_type=emb.get("type", "hierarchical"),
            fine_clusters=emb.get("fine_clusters", 64),
            coarse_clusters=emb.get("coarse_clusters", 8),
            gate_min=emb.get("gate_min", 0.1),
            alpha_init=emb.get("alpha_init", 0.1),
            beta_init=emb.get("beta_init", 0.05),
            use_gate=emb.get("use_gate", True),
            use_hierarchy=emb.get("use_hierarchy", True),
            n_layers=tfm.get("n_layers", 8),
            n_heads=tfm.get("n_heads", 6),
            d_ff=tfm.get("d_ff", 1536),
            dropout=tfm.get("dropout", 0.1),
            activation=tfm.get("activation", "swiglu"),
            norm=tfm.get("norm", "rmsnorm"),
            causal=tfm.get("causal", False),
            tie_weights=head.get("tie_weights", True),
        )
