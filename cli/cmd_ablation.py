"""z86 ablation — run and compare ablation studies and version configs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from cli import ui


# Legacy ablation configs (quick component tests)
ABLATIONS = [
    ("A0", "flat_autoregressive", "Standard AR baseline"),
    ("A1", "flat_baseline", "Diffusion + flat embedding"),
    ("A2", "clusters_only", "Diffusion + fine clusters, no hierarchy"),
    ("A3", "no_gate", "No gating mechanism"),
    ("A4", "no_meta", "No meta-optimizer"),
    ("A5", "full", "Full HCLM-D (everything enabled)"),
]

# Versioned configs (progressive comparison chain)
VERSIONS = [
    ("v0", "v0_flat_ar", "flat-AR", "Autoregressive baseline"),
    ("v1", "v1_flat_diffusion", "flat-diff", "Masked diffusion, no structure"),
    ("v2", "v2_clusters_only", "clust-only", "Fine clusters (K=64), no hierarchy"),
    ("v3", "v3_hier_naive", "hier-naive", "Full hierarchy, static lambdas (weak)"),
    ("v4", "v4_hier_boost", "hier-boost", "Hierarchy + temp annealing + loss curriculum"),
    ("v5", "v5_hier_boost_meta", "hier-boost+meta", "v4 + meta-optimizer"),
]


def cmd_ablation(args):
    """Run or inspect ablation studies."""
    if args.sub == "run":
        return _run(args)
    elif args.sub == "status":
        return _status(args)
    elif args.sub == "compare":
        return _compare(args)
    elif args.sub == "matrix":
        return _matrix(args)
    else:
        ui.err(f"Unknown subcommand: {args.sub}")
        ui.info("Usage: z86 ablation {run|status|compare|matrix}")
        return 1


def _run(args):
    """Run all (or specific) ablation/version configs."""
    ui.logo()

    # Determine which set to run
    use_versions = getattr(args, "versions", False)
    entries = VERSIONS if use_versions else ABLATIONS

    if use_versions:
        ui.step("Running version comparison chain")
    else:
        ui.step("Running ablation studies")

    targets = entries
    if args.only:
        only_lower = [x.lower() for x in args.only]
        if use_versions:
            targets = [e for e in entries if e[0].lower() in only_lower or e[2].lower() in only_lower]
        else:
            targets = [e for e in entries if e[0].lower() in only_lower or e[1] in args.only]
        if not targets:
            ui.err(f"No matching entries: {args.only}")
            return 1

    results = []
    for entry in targets:
        if use_versions:
            tag_id, name, tag, desc = entry
            config = f"configs/versions/{name}.yaml"
        else:
            tag_id, name, desc = entry
            tag = tag_id
            config = f"configs/ablations/{name}.yaml"

        if not Path(config).exists():
            ui.warn(f"Config not found: {config} — skipping {tag_id}")
            results.append((tag_id, tag, name, "skip"))
            continue

        ui.step(f"{tag_id} [{tag}]: {desc}")
        ui.info(f"Config: {config}")

        ret = subprocess.run(
            [sys.executable, "scripts/train.py", "--config", config],
        ).returncode

        status = "pass" if ret == 0 else "fail"
        results.append((tag_id, tag if use_versions else tag_id, name, status))

        if status == "pass":
            ui.ok(f"{tag_id} completed")
        else:
            ui.err(f"{tag_id} failed (exit {ret})")

    # Summary
    ui.header("Summary")
    headers = ["ID", "TAG", "CONFIG", "STATUS"]
    rows = []
    for tag_id, tag, name, status in results:
        s = (f"{ui.C.GREEN}PASS{ui.C.RST}" if status == "pass"
             else f"{ui.C.RED}FAIL{ui.C.RST}" if status == "fail"
             else f"{ui.C.YELLOW}SKIP{ui.C.RST}")
        rows.append([tag_id, tag, name, s])
    ui.table(headers, rows)
    return 0


def _status(args):
    """Show which ablation/version checkpoints exist."""
    ui.logo()
    ui.step("Ablation & version status")

    headers = ["ID", "TAG", "CONFIG", "EXISTS", "DESCRIPTION"]
    rows = []

    ui.info("── Versions (progressive chain) ──")
    for tag_id, name, tag, desc in VERSIONS:
        config = f"configs/versions/{name}.yaml"
        exists = Path(config).exists()
        config_str = f"{ui.C.GREEN}✓{ui.C.RST}" if exists else f"{ui.C.RED}✗{ui.C.RST}"
        rows.append([tag_id, f"{ui.C.CYAN}{tag}{ui.C.RST}", name, config_str, desc])

    rows.append(["", "", "", "", ""])

    for tag_id, name, desc in ABLATIONS:
        config = f"configs/ablations/{name}.yaml"
        exists = Path(config).exists()
        config_str = f"{ui.C.GREEN}✓{ui.C.RST}" if exists else f"{ui.C.RED}✗{ui.C.RST}"
        rows.append([tag_id, f"{ui.C.GRAY}{tag_id}{ui.C.RST}", name, config_str, desc])

    ui.table(headers, rows)
    return 0


def _compare(args):
    """Compare ablation results (delegates to dashboard or local)."""
    ui.logo()
    ui.step("Ablation comparison")
    ui.info("Use the dashboard ablations page for visual comparison:")
    ui.info("  z86 dashboard")
    ui.info("  Open http://localhost:3000/ablations")
    ui.info("")
    ui.info("Or use z86 diff to compare two versions:")
    ui.info("  z86 diff flat-diff hier-boost")
    return 0


def _matrix(args):
    """Show the comparison matrix — what each version tests."""
    ui.logo()
    ui.header("Version Comparison Matrix")

    features = ["Diffusion", "Causal", "Clusters", "Hierarchy", "Gate",
                 "Temp Anneal", "Loss Curric", "Stagger", "Meta"]

    #                    diff  causal clust hier  gate  temp  loss  stag  meta
    matrix = {
        "v0 flat-AR":       ["✗", "✓", "✗", "✗", "✗", "✗", "✗", "✗", "✗"],
        "v1 flat-diff":     ["✓", "✗", "✗", "✗", "✗", "✗", "✗", "✗", "✗"],
        "v2 clust-only":    ["✓", "✗", "✓", "✗", "✓", "✓", "✓", "✗", "✗"],
        "v3 hier-naive":    ["✓", "✗", "✓", "✓", "✓", "✗", "✗", "✗", "✗"],
        "v4 hier-boost":    ["✓", "✗", "✓", "✓", "✓", "✓", "✓", "✓", "✗"],
        "v5 hier-boost+m":  ["✓", "✗", "✓", "✓", "✓", "✓", "✓", "✓", "✓"],
    }

    headers = ["VERSION"] + features
    rows = []
    for name, checks in matrix.items():
        colored = []
        for c in checks:
            if c == "✓":
                colored.append(f"{ui.C.GREEN}✓{ui.C.RST}")
            else:
                colored.append(f"{ui.C.GRAY}·{ui.C.RST}")
        rows.append([name] + colored)

    ui.table(headers, rows)

    print()
    ui.info("Key comparisons:")
    ui.info("  v1 vs v0  → Does diffusion beat AR?")
    ui.info("  v2 vs v1  → Do clusters add value?")
    ui.info("  v3 vs v2  → Does hierarchy add value (naive)?")
    ui.info("  v4 vs v3  → Do proposals A+B+D fix the weak hierarchy?")
    ui.info("  v4 vs v1  → Total value of structured embeddings")
    ui.info("  v5 vs v4  → Does meta-optimizer help?")
    return 0
