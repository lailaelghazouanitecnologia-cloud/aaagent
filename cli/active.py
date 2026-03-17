"""Active version — tracks which version config is selected for training.

Stores the active selection in .z86_active (project root).
"""

from __future__ import annotations

import json
from pathlib import Path

ACTIVE_FILE = Path(".z86_active")

# Maps version keys to their config files
VERSION_CONFIGS = {
    "v0": ("configs/versions/v0_flat_ar.yaml", "flat-AR"),
    "v1": ("configs/versions/v1_flat_diffusion.yaml", "flat-diff"),
    "v1-fast": ("configs/versions/v1_flat_diffusion_fast.yaml", "flat-diff-fast"),
    "v2": ("configs/versions/v2_clusters_only.yaml", "clust-only"),
    "v3": ("configs/versions/v3_hier_naive.yaml", "hier-naive"),
    "v4": ("configs/versions/v4_hier_boost.yaml", "hier-boost"),
    "v4-fast": ("configs/versions/v4_hier_boost_fast.yaml", "hier-boost-fast"),
    "v5": ("configs/versions/v5_hier_boost_meta.yaml", "hier-boost+meta"),
}

# Also resolve by tag
TAG_TO_KEY = {tag: key for key, (_, tag) in VERSION_CONFIGS.items()}


def resolve_version(name: str) -> tuple[str, str, str] | None:
    """Resolve a version name to (key, config_path, tag).

    Accepts: "v1", "v4-fast", "flat-diff", "hier-boost", config path.
    """
    # Direct key match
    if name in VERSION_CONFIGS:
        config, tag = VERSION_CONFIGS[name]
        return name, config, tag

    # Tag match
    name_lower = name.lower()
    if name_lower in TAG_TO_KEY:
        key = TAG_TO_KEY[name_lower]
        config, tag = VERSION_CONFIGS[key]
        return key, config, tag

    # Config path
    if Path(name).exists() and name.endswith(".yaml"):
        return name, name, ""

    return None


def set_active(key: str, config: str, tag: str) -> None:
    """Set the active version."""
    ACTIVE_FILE.write_text(json.dumps({
        "key": key,
        "config": config,
        "tag": tag,
    }, indent=2))


def get_active() -> tuple[str, str, str] | None:
    """Get the active version: (key, config_path, tag) or None."""
    if not ACTIVE_FILE.exists():
        return None
    try:
        data = json.loads(ACTIVE_FILE.read_text())
        return data["key"], data["config"], data["tag"]
    except (json.JSONDecodeError, KeyError):
        return None


def clear_active() -> None:
    """Clear the active version."""
    if ACTIVE_FILE.exists():
        ACTIVE_FILE.unlink()
