"""z86 set — select which version to train.

Usage:
    z86 set v1              Set flat diffusion as active
    z86 set v4-fast         Set hierarchy boost (fast) as active
    z86 set hier-boost      Set by tag name
    z86 set                 Show current active version + available list
    z86 set --clear         Clear active selection

After setting:
    z86 train               Trains the active version (no --config needed)
    z86 eval                Evals the active version
"""

from __future__ import annotations

from cli import ui
from cli.active import (
    VERSION_CONFIGS, resolve_version, set_active, get_active, clear_active,
)


def cmd_set(args):
    """Set or show the active version."""
    ui.logo()

    # Clear
    if getattr(args, "clear", False):
        clear_active()
        ui.ok("Active version cleared")
        return 0

    # No argument: show status + list
    if not args.version:
        return _show_status()

    # Set
    result = resolve_version(args.version)
    if result is None:
        ui.err(f"Unknown version: {args.version}")
        print()
        _show_available()
        return 1

    key, config, tag = result
    set_active(key, config, tag)

    ui.ok(f"Active version: {key} [{tag}]")
    ui.info(f"Config: {config}")
    print()
    ui.info("Now you can just run:")
    ui.info(f"  z86 train               → trains [{tag}]")
    ui.info(f"  z86 train --resume v3   → resume from v3 with [{tag}] config")
    ui.info(f"  z86 eval                → eval latest [{tag}] checkpoint")
    return 0


def _show_status():
    """Show current active version and available list."""
    active = get_active()

    if active:
        key, config, tag = active
        ui.step(f"Active: {key} [{tag}]")
        ui.info(f"Config: {config}")
    else:
        ui.info("No active version set.")
        ui.info("Set one with: z86 set <version>")

    print()
    _show_available()
    return 0


def _show_available():
    """Show available versions."""
    active = get_active()
    active_key = active[0] if active else None

    ui.header("Available versions")
    headers = ["KEY", "TAG", "CONFIG", ""]
    rows = []

    for key, (config, tag) in sorted(VERSION_CONFIGS.items()):
        marker = f"{ui.C.GREEN}← active{ui.C.RST}" if key == active_key else ""
        rows.append([key, tag, config, marker])

    ui.table(headers, rows)
    print()
    ui.info("Usage: z86 set v4-fast")
