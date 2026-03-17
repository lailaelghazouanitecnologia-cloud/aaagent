"""z86 init — setup project, verify deps, prepare data.
   z86 doctor — check environment health."""

from __future__ import annotations

import os
import sys
import shutil
import subprocess
from pathlib import Path

from cli import ui


def cmd_init(args):
    """Full project setup: deps + data + smoke test."""
    ui.logo()
    ui.step("Initializing HCLM-D project")

    # 1. Python version
    v = sys.version_info
    if v >= (3, 10):
        ui.ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        ui.err(f"Python {v.major}.{v.minor} — need 3.10+")
        return 1

    # 2. Check torch
    try:
        import torch
        cuda_str = f"CUDA {torch.version.cuda}" if torch.cuda.is_available() else "CPU only"
        ui.ok(f"torch {torch.__version__} ({cuda_str})")
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            ui.ok(f"GPU: {name} ({mem:.0f}GB)")
        else:
            ui.warn("No GPU detected — training will be very slow")
    except ImportError:
        ui.err("torch not installed — run: pip install -e '.[dev]'")
        return 1

    # 3. Install package
    ui.step("Installing dependencies")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".[dev]", "-q"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        ui.ok("Dependencies installed")
    else:
        ui.err(f"pip install failed: {result.stderr[:200]}")
        return 1

    # 4. Data preparation
    data_dir = Path("data/tinystories")
    tok_path = Path("data/tokenizer.json")

    if (data_dir / "train_tokens.pt").exists() and tok_path.exists():
        ui.ok("Data already prepared")
    else:
        ui.step("Preparing data (download + tokenize)")
        ui.info("This may take ~30 minutes on first run...")
        result = subprocess.run(
            [sys.executable, "scripts/train.py", "--config", "configs/base.yaml", "--phase", "prep"],
            capture_output=False,
        )
        if result.returncode == 0:
            ui.ok("Data prepared")
        else:
            ui.err("Data preparation failed")
            return 1

    # 5. Smoke test
    ui.step("Running smoke test")
    result = subprocess.run(
        [sys.executable, "scripts/smoke_test.py"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        ui.ok("Smoke test passed")
    else:
        ui.warn("Smoke test had issues — check scripts/smoke_test.py")

    # 6. Dashboard check
    if shutil.which("bun"):
        ui.ok("Bun available (for dashboard)")
    else:
        ui.info("Bun not found — dashboard requires: curl -fsSL https://bun.sh/install | bash")

    ui.header("Ready")
    ui.info("Start training:  z86 train")
    ui.info("Start dashboard: z86 dashboard")
    return 0


def cmd_doctor(args):
    """Check environment health."""
    ui.logo()
    ui.step("System check")

    issues = 0

    # Python
    v = sys.version_info
    if v >= (3, 10):
        ui.ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        ui.err(f"Python {v.major}.{v.minor} — need 3.10+")
        issues += 1

    # Torch + CUDA
    try:
        import torch
        ui.ok(f"torch {torch.__version__}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                ui.ok(f"GPU {i}: {name} ({mem:.0f}GB)")
        else:
            ui.warn("No CUDA GPU available")
    except ImportError:
        ui.err("torch not installed")
        issues += 1

    # Key dependencies
    for pkg in ["tokenizers", "pyyaml", "numpy", "tqdm", "datasets", "requests"]:
        try:
            __import__(pkg.replace("-", "_"))
            ui.ok(pkg)
        except ImportError:
            ui.err(f"{pkg} not installed")
            issues += 1

    # Optional
    for pkg, label in [("wandb", "wandb"), ("agno", "agno (eval agent)"), ("groq", "groq (LLM judge)")]:
        try:
            __import__(pkg)
            ui.ok(label)
        except ImportError:
            ui.info(f"{label} — not installed (optional)")

    # Data
    ui.step("Data")
    data_dir = Path("data/tinystories")
    tok_path = Path("data/tokenizer.json")

    if tok_path.exists():
        ui.ok(f"Tokenizer: {tok_path}")
    else:
        ui.warn("Tokenizer not found — run: z86 init")
        issues += 1

    for split in ["train", "val"]:
        p = data_dir / f"{split}_tokens.pt"
        if p.exists():
            size = p.stat().st_size / (1024**3)
            ui.ok(f"{split}: {p} ({size:.1f}GB)")
        else:
            ui.warn(f"{split} tokens not found")
            issues += 1

    # Checkpoints
    ui.step("Checkpoints")
    ckpt_dir = Path("checkpoints")
    if ckpt_dir.exists():
        ckpts = list(ckpt_dir.glob("*.pt"))
        ui.ok(f"{len(ckpts)} checkpoint(s) in {ckpt_dir}")
    else:
        ui.info("No checkpoints yet")

    # Dashboard
    ui.step("Dashboard")
    if shutil.which("bun"):
        ui.ok("Bun available")
        pkg = Path("dashboard/package.json")
        nm = Path("dashboard/node_modules")
        if pkg.exists() and nm.exists():
            ui.ok("Dashboard dependencies installed")
        elif pkg.exists():
            ui.warn("Run: cd dashboard && bun install")
        else:
            ui.err("dashboard/package.json not found")
            issues += 1
    else:
        ui.info("Bun not available (dashboard disabled)")

    # Env vars — basic
    ui.step("Environment")
    for var, desc in [
        ("DASHBOARD_URL", "dashboard URL"),
        ("WANDB_PROJECT", "W&B project"),
    ]:
        val = os.environ.get(var)
        if val:
            ui.ok(f"{var}={val[:30]}{'...' if len(val) > 30 else ''}")
        else:
            ui.info(f"{var} not set ({desc})")

    # ── Cloud APIs (Groq + Cloudflare + RunPod) ──
    ui.step("Cloud APIs")

    # Groq
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        try:
            from cloud.groq import check_api_key as check_groq
            ok, detail = check_groq()
            if ok:
                ui.ok(f"GROQ_API_KEY ✓ ({detail})")
            else:
                ui.warn(f"GROQ_API_KEY set but failed: {detail}")
                issues += 1
        except Exception as e:
            ui.warn(f"GROQ_API_KEY set (connectivity check skipped: {e})")
    else:
        ui.info("GROQ_API_KEY not set (needed for LLM judge eval)")

    # Cloudflare
    cf_token = os.environ.get("CF_API_TOKEN", "")
    cf_zone = os.environ.get("CF_ZONE_ID", "")
    if cf_token:
        try:
            from cloud.cloudflare import check_api_token
            if check_api_token():
                ui.ok(f"CF_API_TOKEN ✓ (valid)")
            else:
                ui.warn("CF_API_TOKEN set but invalid")
                issues += 1
        except Exception as e:
            ui.warn(f"CF_API_TOKEN set (check skipped: {e})")
        if cf_zone:
            ui.ok(f"CF_ZONE_ID={cf_zone[:20]}...")
        else:
            ui.info("CF_ZONE_ID not set (needed for DNS management)")
    else:
        ui.info("CF_API_TOKEN not set (needed for Cloudflare DNS)")

    # RunPod
    runpod_key = os.environ.get("RUNPOD_API_KEY", "")
    if runpod_key:
        try:
            from cloud.runpod import RunPodClient
            client = RunPodClient(runpod_key)
            pods = client.list_pods()
            active = [p for p in pods if p.status == "RUNNING"]
            ui.ok(f"RUNPOD_API_KEY ✓ ({len(pods)} pods, {len(active)} running)")
        except Exception as e:
            ui.warn(f"RUNPOD_API_KEY set but failed: {e}")
            issues += 1
    else:
        ui.info("RUNPOD_API_KEY not set (needed for cloud GPU)")

    # Summary
    if issues == 0:
        ui.header("All clear")
    else:
        ui.header(f"{issues} issue(s) found")
    return 0 if issues == 0 else 1
