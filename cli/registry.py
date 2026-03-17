"""Version registry — tracks checkpoints as named versions.

Stores manifest at checkpoints/manifest.json with version metadata.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

CHECKPOINTS_DIR = Path("checkpoints")
MANIFEST_PATH = CHECKPOINTS_DIR / "manifest.json"


@dataclass
class Version:
    id: str                     # "v1", "v2", ...
    step: int
    path: str                   # relative to project root
    run: str                    # run name
    config: str                 # config file used
    loss: float | None = None
    ppl: float | None = None
    entropy: float | None = None
    gate_mean: float | None = None
    created: str = ""           # ISO timestamp
    size_mb: float = 0.0
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> Version:
        return Version(**{k: v for k, v in d.items() if k in Version.__dataclass_fields__})


class Registry:
    """Manages the version manifest."""

    def __init__(self, root: Path | str = "."):
        self.root = Path(root)
        self.manifest_path = self.root / MANIFEST_PATH
        self.versions: list[Version] = []
        self._load()

    def _load(self):
        if self.manifest_path.exists():
            data = json.loads(self.manifest_path.read_text())
            self.versions = [Version.from_dict(v) for v in data.get("versions", [])]

    def _save(self):
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"versions": [v.to_dict() for v in self.versions]}
        self.manifest_path.write_text(json.dumps(data, indent=2))

    def next_id(self) -> str:
        """Return next version id (v1, v2, ...)."""
        if not self.versions:
            return "v1"
        nums = []
        for v in self.versions:
            try:
                nums.append(int(v.id.lstrip("v")))
            except ValueError:
                pass
        return f"v{max(nums, default=0) + 1}"

    def get(self, version_id: str) -> Version | None:
        """Get version by id."""
        for v in self.versions:
            if v.id == version_id:
                return v
        return None

    def latest(self) -> Version | None:
        """Get the most recent version."""
        if not self.versions:
            return None
        return self.versions[-1]

    def best(self) -> Version | None:
        """Get the version with lowest loss."""
        candidates = [v for v in self.versions if v.loss is not None]
        if not candidates:
            return None
        return min(candidates, key=lambda v: v.loss)

    def list_all(self) -> list[Version]:
        return list(self.versions)

    def register(
        self,
        step: int,
        path: str,
        run: str,
        config: str,
        loss: float | None = None,
        ppl: float | None = None,
        entropy: float | None = None,
        gate_mean: float | None = None,
        note: str = "",
        version_id: str | None = None,
    ) -> Version:
        """Register a new version from a checkpoint."""
        vid = version_id or self.next_id()

        # Check if path exists and get size
        full_path = self.root / path
        size_mb = full_path.stat().st_size / (1024 * 1024) if full_path.exists() else 0.0

        v = Version(
            id=vid,
            step=step,
            path=path,
            run=run,
            config=config,
            loss=loss,
            ppl=ppl,
            entropy=entropy,
            gate_mean=gate_mean,
            created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            size_mb=round(size_mb, 1),
            note=note,
        )
        self.versions.append(v)
        self._save()
        return v

    def delete(self, version_id: str, delete_file: bool = False) -> bool:
        """Remove a version from the registry."""
        v = self.get(version_id)
        if v is None:
            return False

        if delete_file:
            p = self.root / v.path
            if p.exists():
                p.unlink()

        self.versions = [x for x in self.versions if x.id != version_id]
        self._save()
        return True

    def update(self, version_id: str, **kwargs) -> bool:
        """Update version metadata."""
        v = self.get(version_id)
        if v is None:
            return False
        for k, val in kwargs.items():
            if hasattr(v, k):
                setattr(v, k, val)
        self._save()
        return True

    def scan_unregistered(self) -> list[Path]:
        """Find checkpoint files not in the registry."""
        ckpt_dir = self.root / CHECKPOINTS_DIR
        if not ckpt_dir.exists():
            return []

        registered_paths = {v.path for v in self.versions}
        unregistered = []
        for f in sorted(ckpt_dir.glob("*.pt")):
            rel = str(f.relative_to(self.root))
            if rel not in registered_paths:
                unregistered.append(f)
        return unregistered
