"""z86 storage — configure where checkpoints, metrics, and data live.

When training on RunPod with a network volume mounted at /workspace,
this command configures all paths to persist on the volume so nothing
is lost when the pod stops.

Usage:
    z86 storage                       Show current paths
    z86 storage --volume VOL_ID       Set RunPod volume + switch paths to /workspace
    z86 storage --local                Switch back to local paths
    z86 storage --sync-to-volume      Copy local artifacts → /workspace
    z86 storage --sync-from-volume    Copy /workspace artifacts → local
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from cli import ui

STORAGE_FILE = Path(".z86_storage")
WORKSPACE = Path("/workspace")

# Default paths (local)
LOCAL_PATHS = {
    "checkpoints": "checkpoints",
    "metrics": "metrics",
    "data": "data/tinystories",
    "results": "results",
    "manifest": "checkpoints/manifest.json",
}

# Volume paths (RunPod /workspace)
VOLUME_PATHS = {
    "checkpoints": "/workspace/checkpoints",
    "metrics": "/workspace/metrics",
    "data": "/workspace/data/tinystories",
    "results": "/workspace/results",
    "manifest": "/workspace/checkpoints/manifest.json",
}

ARTIFACTS = ["checkpoints", "metrics", "results"]


def cmd_storage(args):
    """Configure storage paths."""
    ui.logo()

    if getattr(args, "volume", None):
        return _set_volume(args.volume)
    if getattr(args, "local", False):
        return _set_local()
    if getattr(args, "sync_to_volume", False):
        return _sync(to_volume=True)
    if getattr(args, "sync_from_volume", False):
        return _sync(to_volume=False)

    return _show_status()


def _show_status():
    """Show current storage configuration."""
    cfg = load_storage_config()

    if cfg:
        mode = cfg.get("mode", "local")
        ui.step(f"Storage mode: {mode}")
        if cfg.get("volume_id"):
            ui.info(f"RunPod volume: {cfg['volume_id']}")
        print()
        ui.header("Paths")
        paths = cfg.get("paths", LOCAL_PATHS)
        for name, path in paths.items():
            exists = Path(path).exists()
            status = f"{ui.C.GREEN}✓{ui.C.RST}" if exists else f"{ui.C.GRAY}·{ui.C.RST}"
            ui.info(f"  {status} {name:15s} → {path}")
    else:
        ui.step("Storage mode: local (default)")
        print()
        ui.header("Paths")
        for name, path in LOCAL_PATHS.items():
            exists = Path(path).exists()
            status = f"{ui.C.GREEN}✓{ui.C.RST}" if exists else f"{ui.C.GRAY}·{ui.C.RST}"
            ui.info(f"  {status} {name:15s} → {path}")

    print()
    ui.info("Configure:")
    ui.info("  z86 storage --volume VOL_ID    → persist on RunPod volume")
    ui.info("  z86 storage --local            → use local paths")
    ui.info("  z86 storage --sync-to-volume   → copy local → /workspace")
    return 0


def _set_volume(volume_id: str):
    """Switch to volume-backed storage."""
    # Verify /workspace exists (we're on RunPod)
    if not WORKSPACE.exists():
        ui.warn("/workspace not found — are you on a RunPod pod?")
        ui.info("Setting volume_id anyway for later use.")

    # Create dirs on volume
    for name, path in VOLUME_PATHS.items():
        if name != "manifest":
            Path(path).mkdir(parents=True, exist_ok=True)

    cfg = {
        "mode": "volume",
        "volume_id": volume_id,
        "paths": VOLUME_PATHS,
    }
    _save_config(cfg)

    # Also update config.toml with volume_id
    _update_config_toml_volume(volume_id)

    ui.ok(f"Storage switched to volume: {volume_id}")
    ui.info("All checkpoints, metrics, and results will save to /workspace/")
    ui.info("This data persists across pod stop/start.")
    print()
    ui.info("Symlinks created for transparent access:")
    _create_symlinks()
    return 0


def _set_local():
    """Switch back to local storage."""
    cfg = {
        "mode": "local",
        "volume_id": "",
        "paths": LOCAL_PATHS,
    }
    _save_config(cfg)
    _remove_symlinks()
    ui.ok("Storage switched to local")
    return 0


def _sync(to_volume: bool):
    """Sync artifacts between local and volume."""
    if not WORKSPACE.exists():
        ui.err("/workspace not found. Are you on a RunPod pod?")
        return 1

    for name in ARTIFACTS:
        local = Path(LOCAL_PATHS[name])
        remote = Path(VOLUME_PATHS[name])

        if to_volume:
            src, dst = local, remote
        else:
            src, dst = remote, local

        if not src.exists():
            ui.info(f"  Skip {name}: {src} not found")
            continue

        dst.mkdir(parents=True, exist_ok=True)
        count = 0
        for f in src.rglob("*"):
            if f.is_file():
                rel = f.relative_to(src)
                target = dst / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists() or f.stat().st_mtime > target.stat().st_mtime:
                    shutil.copy2(f, target)
                    count += 1

        direction = "→ volume" if to_volume else "→ local"
        ui.ok(f"  {name}: {count} files {direction}")

    return 0


def _create_symlinks():
    """Create symlinks from local paths to volume paths."""
    for name in ARTIFACTS:
        local = Path(LOCAL_PATHS[name])
        remote = Path(VOLUME_PATHS[name])

        if not remote.exists():
            remote.mkdir(parents=True, exist_ok=True)

        # If local exists and is not already a symlink, back it up
        if local.exists() and not local.is_symlink():
            backup = local.with_name(f"{local.name}_local_backup")
            if not backup.exists():
                local.rename(backup)
                ui.info(f"  Backed up {local} → {backup}")
            else:
                shutil.rmtree(local)

        # Create symlink
        if not local.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            local.symlink_to(remote)
            ui.info(f"  {local} → {remote}")


def _remove_symlinks():
    """Remove symlinks, restore local dirs."""
    for name in ARTIFACTS:
        local = Path(LOCAL_PATHS[name])
        if local.is_symlink():
            local.unlink()
            local.mkdir(parents=True, exist_ok=True)
            ui.info(f"  Removed symlink: {local}")

            # Restore backup if exists
            backup = local.with_name(f"{local.name}_local_backup")
            if backup.exists():
                shutil.rmtree(local)
                backup.rename(local)
                ui.info(f"  Restored from backup: {backup}")


def _update_config_toml_volume(volume_id: str):
    """Update config.toml with the volume_id."""
    toml_path = Path("config.toml")
    if not toml_path.exists():
        return

    content = toml_path.read_text()
    # Uncomment and set volume_id
    if '# volume_id = ""' in content:
        content = content.replace(
            '# volume_id = ""',
            f'volume_id = "{volume_id}"',
        )
    elif 'volume_id = "' in content:
        # Replace existing value
        import re
        content = re.sub(
            r'volume_id = "[^"]*"',
            f'volume_id = "{volume_id}"',
            content,
        )
    toml_path.write_text(content)


def _save_config(cfg: dict):
    STORAGE_FILE.write_text(json.dumps(cfg, indent=2))


def load_storage_config() -> dict | None:
    """Load storage config. Used by other modules to resolve paths."""
    if not STORAGE_FILE.exists():
        return None
    try:
        return json.loads(STORAGE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def get_path(name: str) -> str:
    """Get the configured path for a storage component.

    Usage:
        from cli.cmd_storage import get_path
        ckpt_dir = get_path("checkpoints")  # returns volume or local path
    """
    cfg = load_storage_config()
    if cfg:
        return cfg.get("paths", LOCAL_PATHS).get(name, LOCAL_PATHS.get(name, name))
    return LOCAL_PATHS.get(name, name)
