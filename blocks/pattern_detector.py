"""Pattern Detector — finds recurring block sequences and creates templates.

When the system detects that a sequence of blocks appears frequently,
it creates a new template block that compresses the pattern.

This enables the system to self-extend: new templates emerge from usage
patterns, not from training data.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import torch

from blocks.block import Block, BlockType
from blocks.registry import BlockRegistry
from blocks.composer import BlockComposer

logger = logging.getLogger(__name__)


class PatternDetector:
    """Detects recurring block patterns and creates reusable templates.

    Monitors sequences of block hashes, identifies frequent subsequences,
    and composes them into new template blocks for reuse.
    """

    def __init__(
        self,
        registry: BlockRegistry,
        composer: BlockComposer,
        min_frequency: int = 5,
        min_pattern_len: int = 2,
        max_pattern_len: int = 6,
    ):
        self.registry = registry
        self.composer = composer
        self.min_frequency = min_frequency
        self.min_pattern_len = min_pattern_len
        self.max_pattern_len = max_pattern_len

        # Track sequences of block hashes
        self._pattern_counts: dict[tuple[bytes, ...], int] = defaultdict(int)
        self._created_patterns: set[tuple[bytes, ...]] = set()

    def observe(self, block_sequence: list[Block]) -> None:
        """Record a sequence of blocks for pattern detection.

        Extracts all subsequences of length [min, max] and counts them.
        """
        hashes = [b.semantic_hash() for b in block_sequence]

        for length in range(self.min_pattern_len, min(self.max_pattern_len + 1, len(hashes) + 1)):
            for start in range(len(hashes) - length + 1):
                pattern = tuple(hashes[start:start + length])
                self._pattern_counts[pattern] += 1

    def detect_and_create(self) -> list[Block]:
        """Check for frequent patterns and create template blocks.

        Returns list of newly created template blocks.
        """
        new_templates = []

        for pattern, count in sorted(self._pattern_counts.items(), key=lambda x: -x[1]):
            if count < self.min_frequency:
                continue
            if pattern in self._created_patterns:
                continue

            # Resolve blocks from hashes
            blocks = []
            valid = True
            for h in pattern:
                block = self.registry.lookup(h)
                if block is None:
                    valid = False
                    break
                blocks.append(block)

            if not valid:
                continue

            # Create a new template from this pattern
            name = f"pattern_{len(self._created_patterns)}_{count}x"
            template = self.composer.compose(
                blocks=blocks,
                name=name,
                n_open_slots=0,  # Exact match template; override slots later
            )
            template.block_type = BlockType.TEMPLATE

            self.registry.register(template)
            self._created_patterns.add(pattern)
            new_templates.append(template)

            logger.info(
                "Created template '%s' from %d-block pattern (used %dx): %s",
                name, len(blocks), count,
                " → ".join(b.name or b.hex_hash() for b in blocks),
            )

        return new_templates

    def get_frequent_patterns(self, top_k: int = 10) -> list[tuple[tuple[bytes, ...], int]]:
        """Return top-k most frequent patterns."""
        return sorted(
            self._pattern_counts.items(), key=lambda x: -x[1]
        )[:top_k]

    def reset_counts(self) -> None:
        """Reset pattern counts (keep created patterns)."""
        self._pattern_counts.clear()
