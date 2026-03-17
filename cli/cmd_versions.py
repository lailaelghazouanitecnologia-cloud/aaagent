"""z86 versions — list, compare, and manage model versions.
   z86 diff — compare two versions side by side."""

from __future__ import annotations

from pathlib import Path

from cli import ui
from cli.registry import Registry


def cmd_versions(args):
    """List all registered versions."""
    ui.logo()
    reg = Registry()
    versions = reg.list_all()

    if not versions:
        ui.info("No versions registered yet.")
        ui.info("Train a model to create versions: z86 train")

        # Check for unregistered checkpoints
        unregistered = reg.scan_unregistered()
        if unregistered:
            ui.warn(f"Found {len(unregistered)} unregistered checkpoint(s):")
            for p in unregistered:
                ui.info(f"  {p}")
            ui.info("Register them with: z86 versions --scan")
        return 0

    # Scan for unregistered if --scan
    if args.scan:
        return _scan_and_register(reg)

    best = reg.best()

    headers = ["VERSION", "TAG", "STEP", "LOSS", "PPL", "ENTROPY", "NOTE", "DATE"]
    rows = []
    for v in versions:
        is_best = best and v.id == best.id

        vid = f"{ui.C.GREEN}{v.id}{ui.C.RST}" if is_best else v.id
        if is_best:
            vid += f" {ui.C.GREEN}★{ui.C.RST}"

        tag_str = f"{ui.C.CYAN}{v.tag}{ui.C.RST}" if v.tag else f"{ui.C.GRAY}—{ui.C.RST}"
        loss_str = f"{v.loss:.4f}" if v.loss is not None else "—"
        ppl_str = f"{v.ppl:.2f}" if v.ppl is not None else "—"
        ent_str = f"{v.entropy:.3f}" if v.entropy is not None else "—"
        note_str = v.note[:20] if v.note else "—"
        date_str = v.created[:10] if v.created else "—"

        rows.append([vid, tag_str, str(v.step), loss_str, ppl_str, ent_str, note_str, date_str])

    if args.detail:
        # Extended info
        for v in versions:
            tag_display = f" [{v.tag}]" if v.tag else ""
            ui.header(f"{v.id}{tag_display} — step {v.step}")
            if v.tag:
                ui.kv("Tag", v.tag)
            ui.kv("Path", v.path)
            ui.kv("Run", v.run)
            ui.kv("Config", v.config)
            ui.kv("Loss", f"{v.loss:.4f}" if v.loss is not None else "—")
            ui.kv("Perplexity", f"{v.ppl:.2f}" if v.ppl is not None else "—")
            ui.kv("Entropy", f"{v.entropy:.3f}" if v.entropy is not None else "—")
            ui.kv("Gate Mean", f"{v.gate_mean:.4f}" if v.gate_mean is not None else "—")
            ui.kv("Size", f"{v.size_mb:.0f}MB")
            ui.kv("Created", v.created or "—")
            if v.note:
                ui.kv("Note", v.note)
    else:
        ui.table(headers, rows)

    print()
    ui.info(f"{len(versions)} version(s)")
    return 0


def cmd_diff(args):
    """Compare two versions."""
    ui.logo()
    reg = Registry()

    a = reg.get(args.version_a)
    b = reg.get(args.version_b)

    if not a:
        ui.err(f"Version not found: {args.version_a}")
        return 1
    if not b:
        ui.err(f"Version not found: {args.version_b}")
        return 1

    label_a = f"{a.id} [{a.tag}]" if a.tag else a.id
    label_b = f"{b.id} [{b.tag}]" if b.tag else b.id
    ui.header(f"Comparing {label_a} vs {label_b}")

    metrics = [
        ("Step", a.step, b.step, False, "{:,}"),
        ("Loss", a.loss, b.loss, False, "{:.4f}"),
        ("Perplexity", a.ppl, b.ppl, False, "{:.2f}"),
        ("Entropy", a.entropy, b.entropy, True, "{:.3f}"),
        ("Gate Mean", a.gate_mean, b.gate_mean, True, "{:.4f}"),
    ]

    headers = ["METRIC", a.id, b.id, "DELTA", ""]
    rows = []

    for name, va, vb, higher_better, fmt in metrics:
        sa = fmt.format(va) if va is not None else "—"
        sb = fmt.format(vb) if vb is not None else "—"

        if va is not None and vb is not None:
            d = vb - va
            delta = ui.delta_str(d, higher_is_better=higher_better)
            # Percentage
            if va != 0:
                pct = (d / abs(va)) * 100
                pct_str = f"{ui.C.GRAY}({pct:+.1f}%){ui.C.RST}"
            else:
                pct_str = ""
        else:
            delta = f"{ui.C.GRAY}—{ui.C.RST}"
            pct_str = ""

        rows.append([name, sa, sb, delta, pct_str])

    ui.table(headers, rows, col_widths=[16, 12, 12, 12, 10])
    return 0


def cmd_delete(args):
    """Delete a version."""
    reg = Registry()
    v = reg.get(args.version_id)

    if not v:
        ui.err(f"Version not found: {args.version_id}")
        return 1

    ui.info(f"Version: {v.id} (step {v.step}, {v.size_mb:.0f}MB)")

    if args.keep_file:
        if not ui.confirm(f"Remove {v.id} from registry (keep checkpoint file)?"):
            ui.abort()
        reg.delete(v.id, delete_file=False)
        ui.ok(f"Removed {v.id} from registry (file kept: {v.path})")
    else:
        if not ui.confirm(f"Delete {v.id} and its checkpoint file ({v.path})?"):
            ui.abort()
        reg.delete(v.id, delete_file=True)
        ui.ok(f"Deleted {v.id}")

    return 0


def _scan_and_register(reg: Registry) -> int:
    """Find unregistered checkpoints and offer to register them."""
    unregistered = reg.scan_unregistered()
    if not unregistered:
        ui.ok("All checkpoints are registered")
        return 0

    ui.step(f"Found {len(unregistered)} unregistered checkpoint(s)")

    for p in unregistered:
        size = p.stat().st_size / (1024**2)
        # Try to extract step from filename
        try:
            step_num = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            step_num = 0

        ui.info(f"  {p.name} — step {step_num}, {size:.0f}MB")

        if ui.confirm(f"Register as {reg.next_id()}?"):
            # Try to derive tag from filename
            tag = p.stem.replace("step_", "").replace("_", "-")
            v = reg.register(
                step=step_num,
                path=str(p),
                run="unknown",
                config="configs/base.yaml",
                tag=tag,
                note="scanned",
            )
            ui.ok(f"Registered as {v.id} (tag: {v.tag})")

    return 0
