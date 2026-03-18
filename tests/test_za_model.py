"""Smoke tests for Za v7 model (RWKV backbone)."""

import torch
import pytest

from model.config import ModelConfig
from model.za_model import ZaModel


def _tiny_config(**overrides) -> ModelConfig:
    """Create a minimal v7 config for testing."""
    defaults = dict(
        vocab_size=256,
        embed_dim=64,
        meta_embed_dim=32,
        max_seq_len=32,
        mask_token_id=0,
        fine_clusters=8,
        coarse_clusters=4,
        backbone_type="rwkv",
        rwkv_layers=2,
        n_heads=4,
        d_ff=128,
        share_weights=False,
        tie_weights=True,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


class TestZaModelRWKV:
    """Test Za v7 with RWKV backbone."""

    def test_forward_basic(self):
        config = _tiny_config()
        model = ZaModel(config)
        x = torch.randint(0, config.vocab_size, (2, 16))
        logits = model(x)
        assert logits.shape == (2, 16, config.vocab_size)

    def test_forward_with_mask(self):
        config = _tiny_config()
        model = ZaModel(config)
        x = torch.randint(0, config.vocab_size, (2, 16))
        mask = torch.ones(2, 16)
        mask[:, -4:] = 0  # Pad last 4
        logits = model(x, attention_mask=mask)
        assert logits.shape == (2, 16, config.vocab_size)

    def test_forward_with_warmup_params(self):
        config = _tiny_config()
        model = ZaModel(config)
        x = torch.randint(0, config.vocab_size, (2, 16))
        logits = model(
            x,
            gate_scale=torch.tensor(0.5),
            router_temperature=torch.tensor(0.8),
            coarse_temperature=torch.tensor(0.5),
        )
        assert logits.shape == (2, 16, config.vocab_size)

    def test_weight_tying(self):
        config = _tiny_config(tie_weights=True)
        model = ZaModel(config)
        emb_weight = model.embedding.weight
        assert model.head._tied_weight is emb_weight

    def test_param_count(self):
        config = _tiny_config()
        model = ZaModel(config)
        counts = model.count_parameters()
        assert counts["total"] > 0
        assert counts["embedding"] > 0
        assert counts["backbone"] > 0
        assert counts["head"] == 0  # Weight tied

    def test_shared_weights(self):
        config_shared = _tiny_config(share_weights=True)
        config_full = _tiny_config(share_weights=False)
        model_shared = ZaModel(config_shared)
        model_full = ZaModel(config_full)
        shared_params = sum(p.numel() for p in model_shared.parameters())
        full_params = sum(p.numel() for p in model_full.parameters())
        assert shared_params < full_params

    def test_gradient_flows(self):
        config = _tiny_config()
        model = ZaModel(config)
        x = torch.randint(0, config.vocab_size, (2, 8))
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        # Check that RWKV params got gradients
        has_grad = False
        for name, p in model.named_parameters():
            if "time_mix" in name and p.grad is not None:
                has_grad = True
                break
        assert has_grad, "RWKV time_mix params should have gradients"


class TestZaModelTransformer:
    """Test Za v7 with Transformer backbone (ablation)."""

    def test_forward_transformer(self):
        config = _tiny_config(
            backbone_type="transformer",
            transformer_layers=2,
            transformer_heads=4,
            transformer_d_ff=128,
        )
        model = ZaModel(config)
        x = torch.randint(0, config.vocab_size, (2, 16))
        logits = model(x)
        assert logits.shape == (2, 16, config.vocab_size)


class TestRWKVBackbone:
    """Test RWKV backbone directly."""

    def test_bidirectional_output_differs(self):
        """Forward and backward should produce different representations."""
        from model.rwkv.backbone import RWKVBackbone
        backbone = RWKVBackbone(n_layers=2, embed_dim=64, n_heads=4, d_ff=128)
        x = torch.randn(2, 16, 64)
        out = backbone(x)
        assert out.shape == (2, 16, 64)
        # Output should not be all zeros
        assert out.abs().sum() > 0

    def test_padding_mask_zeros(self):
        """Padded positions should be zeroed out."""
        from model.rwkv.backbone import RWKVBackbone
        backbone = RWKVBackbone(n_layers=2, embed_dim=64, n_heads=4, d_ff=128)
        x = torch.randn(2, 16, 64)
        mask = torch.ones(2, 16)
        mask[0, 12:] = 0  # Pad last 4 for batch 0
        out = backbone(x, attention_mask=mask)
        # Padded positions should be zero
        assert out[0, 12:].abs().sum() == 0
        # Non-padded should not be zero
        assert out[0, :12].abs().sum() > 0


class TestV7Config:
    """Test v7 config parsing."""

    def test_from_dict(self):
        d = {
            "model": {
                "vocab_size": 32868,
                "embed_dim": 512,
                "backbone": {"type": "rwkv"},
                "rwkv": {"layers": 4, "n_heads": 8, "d_ff": 2048},
                "embedding": {"fine_clusters": 128, "coarse_clusters": 16},
            }
        }
        cfg = ModelConfig.from_dict(d)
        assert cfg.vocab_size == 32868
        assert cfg.embed_dim == 512
        assert cfg.backbone_type == "rwkv"
        assert cfg.rwkv_layers == 4
        assert cfg.n_heads == 8
        assert cfg.fine_clusters == 128
        assert cfg.coarse_clusters == 16

    def test_n_layers_property(self):
        cfg = _tiny_config(rwkv_layers=4)
        assert cfg.n_layers == 8  # 4 fwd + 4 bwd
