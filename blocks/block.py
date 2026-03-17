"""Block — the fundamental structural unit of Za.

A Block is a node in a hierarchical representation tree.
Everything is a block: tokens (leaves), templates (with slots),
plans (fully-filled trees), meta-blocks (self-describing).

Block = {
    hash:       bytes32          # Deterministic identifier
    embedding:  float[embed_dim] # Vector representation
    slots:      [Block | None]   # Sub-blocks (templates have empty slots)
    mask:       bits[]           # Which slots are filled vs empty
    meta:       float[meta_dim]  # Meta-embedding describing the block itself
    block_type: str              # "token", "template", "plan", "meta"
}
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch


class BlockType(Enum):
    """Type of block in the hierarchy."""
    TOKEN = "token"           # Leaf: single token, no slots
    SPAN = "span"             # Contiguous token sequence
    TEMPLATE = "template"     # Has slots (some filled, some empty)
    PLAN = "plan"             # Fully-filled tree of blocks
    META = "meta"             # Describes another block


@dataclass
class Block:
    """A structural unit in the Za block hierarchy.

    Blocks form trees. A token is a leaf block (no slots).
    A template is a block with slots (some [MASK]).
    A plan is a fully-resolved tree of blocks.
    """
    block_type: BlockType
    embedding: torch.Tensor                          # [embed_dim]
    slots: list[Optional[Block]] = field(default_factory=list)
    meta_embedding: Optional[torch.Tensor] = None    # [meta_dim]
    name: str = ""
    depth: int = 0
    token_id: Optional[int] = None                   # Only for TOKEN blocks

    # Computed lazily
    _semantic_hash: Optional[bytes] = field(default=None, repr=False)
    _instance_hash: Optional[bytes] = field(default=None, repr=False)

    @property
    def mask(self) -> list[bool]:
        """Bitmask: True where slots are filled, False where empty."""
        return [s is not None for s in self.slots]

    @property
    def n_slots(self) -> int:
        return len(self.slots)

    @property
    def n_filled(self) -> int:
        return sum(1 for s in self.slots if s is not None)

    @property
    def n_empty(self) -> int:
        return sum(1 for s in self.slots if s is None)

    @property
    def is_leaf(self) -> bool:
        return len(self.slots) == 0

    @property
    def is_complete(self) -> bool:
        """All slots filled (recursively)."""
        if self.n_empty > 0:
            return False
        return all(s.is_complete for s in self.slots if s is not None)

    def semantic_hash(self) -> bytes:
        """Deterministic hash based on structure and semantics.

        H_sem = sha256(type + embedding_bytes + slot_hashes + mask)
        """
        if self._semantic_hash is not None:
            return self._semantic_hash

        h = hashlib.sha256()
        h.update(self.block_type.value.encode())
        h.update(self.embedding.detach().cpu().numpy().tobytes())

        for slot in self.slots:
            if slot is not None:
                h.update(slot.semantic_hash())
            else:
                h.update(b"\x00" * 32)  # Empty slot marker

        mask_bytes = bytes(self.mask)
        h.update(mask_bytes)

        self._semantic_hash = h.digest()
        return self._semantic_hash

    def instance_hash(self, parent_hash: bytes = b"", position: int = 0) -> bytes:
        """Hash that distinguishes this specific instance.

        H_inst = sha256(H_sem + parent_hash + position + depth)
        """
        if self._instance_hash is not None:
            return self._instance_hash

        h = hashlib.sha256()
        h.update(self.semantic_hash())
        h.update(parent_hash)
        h.update(position.to_bytes(4, "big"))
        h.update(self.depth.to_bytes(4, "big"))

        self._instance_hash = h.digest()
        return self._instance_hash

    def hex_hash(self) -> str:
        """Short hex representation of semantic hash."""
        return self.semantic_hash().hex()[:12]

    def flatten_tokens(self) -> list[int]:
        """Recursively collect all token IDs in order."""
        if self.block_type == BlockType.TOKEN and self.token_id is not None:
            return [self.token_id]
        tokens = []
        for slot in self.slots:
            if slot is not None:
                tokens.extend(slot.flatten_tokens())
        return tokens

    def all_blocks(self) -> list[Block]:
        """DFS traversal of all blocks in the tree."""
        result = [self]
        for slot in self.slots:
            if slot is not None:
                result.extend(slot.all_blocks())
        return result

    def max_depth(self) -> int:
        """Maximum nesting depth."""
        if not self.slots:
            return 0
        child_depths = [s.max_depth() for s in self.slots if s is not None]
        return 1 + max(child_depths) if child_depths else 1

    def __repr__(self) -> str:
        name_str = f" '{self.name}'" if self.name else ""
        slots_str = f" slots={self.n_filled}/{self.n_slots}" if self.slots else ""
        return f"Block({self.block_type.value}{name_str}{slots_str} hash={self.hex_hash()})"
