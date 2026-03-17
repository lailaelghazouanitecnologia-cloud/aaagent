"""z86 generate — generate text from a model version.
   z86 serve — start HTTP inference server."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from cli import ui
from cli.registry import Registry


def cmd_generate(args):
    """Generate text from a checkpoint."""
    ui.logo()
    reg = Registry()

    # Resolve checkpoint
    checkpoint = _resolve(args.version, reg)
    if not checkpoint:
        return 1

    prompt = args.prompt or "Once upon a time"

    if args.interactive:
        return _interactive(checkpoint, args)

    ui.step(f"Generating from {args.version or 'latest'}")
    ui.info(f"Prompt: {prompt}")
    print()

    cmd = [
        sys.executable, "scripts/generate.py",
        "--checkpoint", checkpoint,
        "--prompt", prompt,
        "--seq_len", str(args.seq_len),
        "--steps", str(args.steps),
        "--temperature", str(args.temperature),
        "--n_samples", str(args.n_samples),
    ]

    if args.config:
        cmd += ["--config", args.config]

    return subprocess.run(cmd).returncode


def _interactive(checkpoint: str, args):
    """Interactive REPL mode."""
    ui.step("Interactive mode (Ctrl+C to exit)")
    print()

    # Preload model once
    sys.path.insert(0, ".")
    from scripts.train import load_config
    from model.config import ModelConfig
    from model.lm import HCLMD
    from data.tokenizer import load_tokenizer
    from training.checkpointing import load_checkpoint as load_ckpt
    from eval.generation import generate_samples

    import torch

    config_path = args.config or "configs/base.yaml"
    config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_config = ModelConfig.from_dict(config)
    model = HCLMD(model_config)
    load_ckpt(checkpoint, model, device=device)
    model = model.to(device)
    model.eval()

    tok_path = config.get("data", {}).get("tokenizer_path", "data/tokenizer.json")
    tokenizer = load_tokenizer(tok_path)

    ui.ok("Model loaded")
    print()

    while True:
        try:
            prompt = input(f"  {ui.C.CYAN}>{ui.C.RST} ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("exit", "quit", "q"):
                break

            results = generate_samples(
                model, tokenizer, [prompt],
                seq_len=args.seq_len,
                sampling_steps=args.steps,
                temperature=args.temperature,
                device=device,
            )

            for r in results:
                print(f"\n  {ui.C.WHITE}{r['generated']}{ui.C.RST}\n")

        except KeyboardInterrupt:
            print()
            break
        except Exception as e:
            ui.err(str(e))

    return 0


def cmd_serve(args):
    """Start HTTP inference server."""
    ui.logo()
    reg = Registry()

    checkpoint = _resolve(args.version, reg)
    if not checkpoint:
        return 1

    ui.step(f"Starting inference server on :{args.port}")
    ui.info(f"Model: {args.version or 'latest'}")
    ui.info(f"Checkpoint: {checkpoint}")

    # Inline simple HTTP server using stdlib
    sys.path.insert(0, ".")

    import json
    import torch
    from http.server import HTTPServer, BaseHTTPRequestHandler

    from scripts.train import load_config
    from model.config import ModelConfig
    from model.lm import HCLMD
    from data.tokenizer import load_tokenizer
    from training.checkpointing import load_checkpoint as load_ckpt
    from eval.generation import generate_samples

    config_path = args.config or "configs/base.yaml"
    config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_config = ModelConfig.from_dict(config)
    model = HCLMD(model_config)
    load_ckpt(checkpoint, model, device=device)
    model = model.to(device)
    model.eval()

    tok_path = config.get("data", {}).get("tokenizer_path", "data/tokenizer.json")
    tokenizer = load_tokenizer(tok_path)

    ui.ok("Model loaded")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == "/generate":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                prompt = body.get("prompt", "Once upon a time")
                temp = body.get("temperature", 0.8)
                seq_len = body.get("seq_len", 256)
                steps = body.get("steps", 64)

                results = generate_samples(
                    model, tokenizer, [prompt],
                    seq_len=seq_len, sampling_steps=steps,
                    temperature=temp, device=device,
                )

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"results": results}).encode())
            else:
                self.send_error(404)

        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok"}).encode())
            else:
                self.send_error(404)

        def log_message(self, fmt, *a):
            ui.info(f"  {a[0]} {a[1]} {a[2]}")

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    ui.ok(f"Serving at http://0.0.0.0:{args.port}")
    ui.info("POST /generate  {\"prompt\": \"...\"}")
    ui.info("GET  /health")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
        ui.warn("Server stopped")
        server.server_close()

    return 0


def _resolve(version_id: str | None, reg: Registry) -> str | None:
    if version_id:
        v = reg.get(version_id)
        if v:
            return v.path
        if version_id == "best":
            v = reg.best()
            if v:
                return v.path
        if Path(version_id).exists():
            return version_id
        ui.err(f"Not found: {version_id}")
        return None

    v = reg.latest()
    if v:
        ui.info(f"Using latest: {v.id}")
        return v.path
    ui.err("No versions found")
    return None
