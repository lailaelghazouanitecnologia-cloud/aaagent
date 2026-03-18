"""Model configuration for Za v7.

Backward compatible: accepts legacy n_layers/n_heads/d_ff kwargs
and maps them to the appropriate transformer/rwkv fields.
"""

from __future__ import annotations


class ModelConfig:
    """Full configuration for Za v7 model."""

    def __init__(
        self,
        # Vocabulary and dimensions
        vocab_size: int = 32868,
        embed_dim: int = 512,
        meta_embed_dim: int = 128,
        max_seq_len: int = 2048,
        mask_token_id: int = 0,
        # Embedding
        embedding_type: str = "hierarchical",
        fine_clusters: int = 128,
        coarse_clusters: int = 16,
        gate_min: float = 0.1,
        alpha_init: float = 0.1,
        beta_init: float = 0.05,
        use_gate: bool = True,
        use_hierarchy: bool = True,
        # Backbone
        backbone_type: str = "rwkv",
        # RWKV
        rwkv_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 2048,
        share_weights: bool = False,
        # Transformer
        transformer_layers: int = 8,
        transformer_heads: int = 8,
        transformer_d_ff: int = 2048,
        dropout: float = 0.0,
        activation: str = "swiglu",
        norm: str = "layernorm",
        causal: bool = False,
        # Head
        tie_weights: bool = True,
        # Legacy compat
        n_layers: int | None = None,
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.meta_embed_dim = meta_embed_dim
        self.max_seq_len = max_seq_len
        self.mask_token_id = mask_token_id

        self.embedding_type = embedding_type
        self.fine_clusters = fine_clusters
        self.coarse_clusters = coarse_clusters
        self.gate_min = gate_min
        self.alpha_init = alpha_init
        self.beta_init = beta_init
        self.use_gate = use_gate
        self.use_hierarchy = use_hierarchy

        self.backbone_type = backbone_type
        self.rwkv_layers = rwkv_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.share_weights = share_weights

        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.transformer_d_ff = transformer_d_ff
        self.dropout = dropout
        self.activation = activation
        self.norm = norm
        self.causal = causal

        self.tie_weights = tie_weights

        # Legacy compat: if n_layers is passed, assume transformer mode
        if n_layers is not None:
            self.backbone_type = "transformer"
            self.transformer_layers = n_layers
            self.transformer_heads = n_heads
            self.transformer_d_ff = d_ff
            self.rwkv_layers = max(1, n_layers // 2)

    @property
    def n_layers(self) -> int:
        """Total backbone layers."""
        if self.backbone_type == "rwkv":
            return self.rwkv_layers * 2
        return self.transformer_layers

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        """Create config from a nested config dictionary."""
        model_cfg = d.get("model", d)
        emb = model_cfg.get("embedding", {})
        backbone = model_cfg.get("backbone", {})
        tfm = model_cfg.get("transformer", {})
        rwkv = model_cfg.get("rwkv", {})
        head = model_cfg.get("head", {})

        return cls(
            vocab_size=model_cfg.get("vocab_size", 32868),
            embed_dim=model_cfg.get("embed_dim", 512),
            meta_embed_dim=model_cfg.get("meta_embed_dim", 128),
            max_seq_len=model_cfg.get("max_seq_len", 2048),
            mask_token_id=model_cfg.get("mask_token_id", 0),
            embedding_type=emb.get("type", "hierarchical"),
            fine_clusters=emb.get("fine_clusters", 128),
            coarse_clusters=emb.get("coarse_clusters", 16),
            gate_min=emb.get("gate_min", 0.1),
            alpha_init=emb.get("alpha_init", 0.1),
            beta_init=emb.get("beta_init", 0.05),
            use_gate=emb.get("use_gate", True),
            use_hierarchy=emb.get("use_hierarchy", True),
            backbone_type=backbone.get("type", model_cfg.get("backbone_type", "rwkv")),
            rwkv_layers=rwkv.get("layers", backbone.get("rwkv_layers", 4)),
            n_heads=rwkv.get("n_heads", backbone.get("n_heads", 8)),
            d_ff=rwkv.get("d_ff", backbone.get("d_ff", 2048)),
            share_weights=rwkv.get("share_weights", backbone.get("share_weights", False)),
            transformer_layers=tfm.get("n_layers", 8),
            transformer_heads=tfm.get("n_heads", 8),
            transformer_d_ff=tfm.get("d_ff", 2048),
            dropout=tfm.get("dropout", 0.0),
            activation=tfm.get("activation", "swiglu"),
            norm=tfm.get("norm", "layernorm"),
            causal=tfm.get("causal", False),
            tie_weights=head.get("tie_weights", True),
        )
