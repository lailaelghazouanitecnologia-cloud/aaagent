"""Block Registry — global store of blocks indexed by semantic hash.

Provides:
- Registration and lookup by hash
- Deduplication (same hash = same block, reuse)
- Cache of execution results by hash
- Pattern frequency tracking
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import torch

from blocks.block import Block

logger = logging.getLogger(__name__)


class BlockRegistry:
    """Global registry of blocks indexed by semantic hash.

    Blocks are stored by their semantic hash. Two blocks with the same
    hash are considered identical and deduplicated.
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._blocks: dict[bytes, Block] = {}
        self._exec_cache: dict[bytes, torch.Tensor] = {}
        self._use_count: dict[bytes, int] = defaultdict(int)
        self._name_index: dict[str, bytes] = {}  # name → hash

    def register(self, block: Block) -> Block:
        """Register a block. Returns existing if already registered."""
        h = block.semantic_hash()

        if h in self._blocks:
            self._use_count[h] += 1
            return self._blocks[h]

        # Evict LRU if at capacity
        if len(self._blocks) >= self.max_size:
            self._evict_least_used()

        self._blocks[h] = block
        self._use_count[h] = 1
        if block.name:
            self._name_index[block.name] = h

        return block

    def lookup(self, h: bytes) -> Optional[Block]:
        """Look up a block by semantic hash."""
        block = self._blocks.get(h)
        if block is not None:
            self._use_count[h] += 1
        return block

    def lookup_by_name(self, name: str) -> Optional[Block]:
        """Look up a block by name."""
        h = self._name_index.get(name)
        if h is not None:
            return self.lookup(h)
        return None

    def cache_result(self, h: bytes, result: torch.Tensor) -> None:
        """Cache execution result for a block."""
        self._exec_cache[h] = result.detach()

    def get_cached_result(self, h: bytes) -> Optional[torch.Tensor]:
        """Get cached execution result."""
        return self._exec_cache.get(h)

    def get_use_count(self, h: bytes) -> int:
        return self._use_count.get(h, 0)

    def most_used(self, top_k: int = 10) -> list[tuple[Block, int]]:
        """Return top-k most used blocks."""
        sorted_hashes = sorted(
            self._use_count.items(), key=lambda x: x[1], reverse=True
        )[:top_k]
        return [
            (self._blocks[h], count)
            for h, count in sorted_hashes
            if h in self._blocks
        ]

    def _evict_least_used(self) -> None:
        """Remove least-used block."""
        if not self._use_count:
            return
        min_hash = min(self._use_count, key=self._use_count.get)
        block = self._blocks.pop(min_hash, None)
        self._use_count.pop(min_hash, None)
        self._exec_cache.pop(min_hash, None)
        if block and block.name:
            self._name_index.pop(block.name, None)

    def clear(self) -> None:
        """Clear all registered blocks and caches."""
        self._blocks.clear()
        self._exec_cache.clear()
        self._use_count.clear()
        self._name_index.clear()

    def __len__(self) -> int:
        return len(self._blocks)

    def __contains__(self, h: bytes) -> bool:
        return h in self._blocks
