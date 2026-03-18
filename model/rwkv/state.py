"""RWKV state serialization for SSD persistence.

RWKV states can be pre-computed for documents and saved to SSD,
enabling "infinite context" by loading pre-computed states.

This module handles serialization/deserialization of RWKV hidden states.
Used during inference (not training).
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import torch


@dataclass
class RWKVState:
    """Serializable RWKV state for a single layer."""
    layer_idx: int
    state_tensor: torch.Tensor  # [H, d] per-head state

    def to_bytes(self) -> bytes:
        """Serialize state to bytes for SSD storage."""
        buf = io.BytesIO()
        torch.save({
            "layer_idx": self.layer_idx,
            "state": self.state_tensor.cpu().half(),  # Save as fp16
        }, buf)
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes, device: str = "cpu") -> RWKVState:
        """Deserialize state from bytes."""
        buf = io.BytesIO(data)
        d = torch.load(buf, map_location=device, weights_only=True)
        return cls(
            layer_idx=d["layer_idx"],
            state_tensor=d["state"].float(),
        )


@dataclass
class RWKVFullState:
    """Full RWKV state across all layers (forward + backward)."""
    forward_states: list[RWKVState]
    backward_states: list[RWKVState]

    def to_bytes(self) -> bytes:
        """Serialize full state."""
        buf = io.BytesIO()
        torch.save({
            "fwd": [s.state_tensor.cpu().half() for s in self.forward_states],
            "bwd": [s.state_tensor.cpu().half() for s in self.backward_states],
        }, buf)
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes, device: str = "cpu") -> RWKVFullState:
        """Deserialize full state."""
        buf = io.BytesIO(data)
        d = torch.load(buf, map_location=device, weights_only=True)
        fwd = [RWKVState(i, s.float()) for i, s in enumerate(d["fwd"])]
        bwd = [RWKVState(i, s.float()) for i, s in enumerate(d["bwd"])]
        return cls(forward_states=fwd, backward_states=bwd)

    def size_bytes(self) -> int:
        """Estimate serialized size."""
        total = 0
        for s in self.forward_states + self.backward_states:
            total += s.state_tensor.numel() * 2  # fp16
        return total
