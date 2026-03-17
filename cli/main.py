#!/usr/bin/env python3
"""z86 — HCLM-D command-line interface.

Usage:
    z86 init                          Setup project (deps + data + test)
    z86 doctor                        Check environment health
    z86 train [--config X] [--name N] Start training
    z86 versions [--detail] [--scan]  List model versions
    z86 diff v1 v2                    Compare two versions
    z86 delete <version>              Delete a version
    z86 eval [version] [--quick]      Evaluate a version
    z86 generate [version] "prompt"   Generate text
    z86 generate [version] -i         Interactive REPL
    z86 serve [version] --port 8080   HTTP inference API
    z86 ablation run|status|compare   Ablation studies
    z86 dashboard [--prod]            Start dashboard
    z86 cloud status                  Cloud resources overview
    z86 cloud gpus                    Available GPUs + pricing
    z86 cloud start [--preset X]      Create RunPod GPU pod
    z86 cloud stop <pod_id>           Stop a pod
    z86 cloud terminate <pod_id>      Destroy a pod
    z86 cloud ssh <pod_id>            Get SSH command
    z86 cloud dns                     Cloudflare DNS records
    z86 cloud dns-set <name> <ip>     Set DNS record
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="z86",
        description="HCLM-D training, versioning, eval, and generation CLI",
    )
    sub = p.add_subparsers(dest="command")

    # ── init ──
    sub.add_parser("init", help="Setup project (deps + data + smoke test)")

    # ── doctor ──
    sub.add_parser("doctor", help="Check environment health")

    # ── train ──
    train_p = sub.add_parser("train", help="Start or resume training")
    train_p.add_argument("--config", default="configs/base.yaml", help="Config path or name")
    train_p.add_argument("--resume", default=None, help="Version id or checkpoint to resume from")
    train_p.add_argument("--name", default=None, help="Run name")
    train_p.add_argument("--dashboard", default=None, help="Dashboard URL")

    # ── versions ──
    ver_p = sub.add_parser("versions", help="List model versions")
    ver_p.add_argument("--detail", action="store_true", help="Show extended info")
    ver_p.add_argument("--scan", action="store_true", help="Scan for unregistered checkpoints")

    # ── diff ──
    diff_p = sub.add_parser("diff", help="Compare two versions")
    diff_p.add_argument("version_a", help="First version id")
    diff_p.add_argument("version_b", help="Second version id")

    # ── delete ──
    del_p = sub.add_parser("delete", help="Delete a version")
    del_p.add_argument("version_id", help="Version to delete")
    del_p.add_argument("--keep-file", action="store_true", help="Keep checkpoint file")

    # ── eval ──
    eval_p = sub.add_parser("eval", help="Evaluate a version → metrics/{tag}/")
    eval_p.add_argument("version", nargs="?", default=None, help="Version id, tag, 'latest', 'best', or path")
    eval_p.add_argument("--config", default=None, help="Config override")
    eval_p.add_argument("--quick", action="store_true", help="Skip perplexity")
    eval_p.add_argument("--judge", action="store_true", help="Enable LLM judge (needs GROQ_API_KEY)")
    eval_p.add_argument("--dashboard", action="store_true", help="Send to dashboard")
    eval_p.add_argument("--prompts", nargs="+", default=None, help="Custom prompts")
    eval_p.add_argument("--compare", nargs=2, metavar=("A", "B"), default=None,
                        help="Compare two versions: --compare flat-diff hier-boost")

    # ── generate ──
    gen_p = sub.add_parser("generate", help="Generate text")
    gen_p.add_argument("version", nargs="?", default=None, help="Version id or path")
    gen_p.add_argument("prompt", nargs="?", default=None, help="Generation prompt")
    gen_p.add_argument("-i", "--interactive", action="store_true", help="Interactive REPL")
    gen_p.add_argument("--config", default=None)
    gen_p.add_argument("--seq_len", type=int, default=256)
    gen_p.add_argument("--steps", type=int, default=64)
    gen_p.add_argument("--temperature", type=float, default=0.8)
    gen_p.add_argument("--n_samples", type=int, default=1)

    # ── serve ──
    serve_p = sub.add_parser("serve", help="Start inference HTTP server")
    serve_p.add_argument("version", nargs="?", default=None, help="Version id or path")
    serve_p.add_argument("--config", default=None)
    serve_p.add_argument("--port", type=int, default=8080)

    # ── ablation ──
    abl_p = sub.add_parser("ablation", help="Ablation studies and version comparisons")
    abl_p.add_argument("sub", choices=["run", "status", "compare", "matrix"], help="Subcommand")
    abl_p.add_argument("--only", nargs="*", default=None, help="Only run specific ablations/versions")
    abl_p.add_argument("--versions", action="store_true", help="Use version configs instead of ablations")

    # ── dashboard ──
    dash_p = sub.add_parser("dashboard", help="Start monitoring dashboard")
    dash_p.add_argument("--port", type=int, default=None, help="Server port")
    dash_p.add_argument("--prod", action="store_true", help="Production mode")

    # ── cloud ──
    cloud_p = sub.add_parser("cloud", help="Manage cloud GPUs (RunPod) and DNS (Cloudflare)")
    cloud_sub = cloud_p.add_subparsers(dest="cloud_sub")

    cloud_sub.add_parser("status", help="Show all cloud resources")
    cloud_sub.add_parser("gpus", help="List available GPU types + pricing")
    cloud_sub.add_parser("volumes", help="List network volumes")

    cloud_start = cloud_sub.add_parser("start", help="Create a new GPU pod")
    cloud_start.add_argument("--preset", choices=["train-small", "train-fast", "inference"],
                             default=None, help="GPU preset")
    cloud_start.add_argument("--name", dest="pod_name", default="hclm-d", help="Pod name")
    cloud_start.add_argument("--volume-id", dest="volume_id", default=None,
                             help="Mount existing network volume (overrides config.toml)")

    cloud_stop = cloud_sub.add_parser("stop", help="Stop a running pod")
    cloud_stop.add_argument("pod_id", help="Pod ID")

    cloud_term = cloud_sub.add_parser("terminate", help="Permanently destroy a pod")
    cloud_term.add_argument("pod_id", help="Pod ID")

    cloud_ssh = cloud_sub.add_parser("ssh", help="Get SSH command for a pod")
    cloud_ssh.add_argument("pod_id", help="Pod ID")

    cloud_sub.add_parser("dns", help="Show Cloudflare DNS records")

    cloud_dns_set = cloud_sub.add_parser("dns-set", help="Create/update DNS record")
    cloud_dns_set.add_argument("dns_name", help="Record name (e.g. z86.dev)")
    cloud_dns_set.add_argument("dns_content", help="Record value (e.g. IP address)")
    cloud_dns_set.add_argument("--type", dest="dns_type", default="A", help="Record type")
    cloud_dns_set.add_argument("--no-proxy", action="store_true", help="Disable Cloudflare proxy")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Lazy import commands to keep startup fast
    if args.command == "init":
        from cli.cmd_init import cmd_init
        return cmd_init(args)

    elif args.command == "doctor":
        from cli.cmd_init import cmd_doctor
        return cmd_doctor(args)

    elif args.command == "train":
        from cli.cmd_train import cmd_train
        return cmd_train(args)

    elif args.command == "versions":
        from cli.cmd_versions import cmd_versions
        return cmd_versions(args)

    elif args.command == "diff":
        from cli.cmd_versions import cmd_diff
        return cmd_diff(args)

    elif args.command == "delete":
        from cli.cmd_versions import cmd_delete
        return cmd_delete(args)

    elif args.command == "eval":
        from cli.cmd_eval import cmd_eval
        return cmd_eval(args)

    elif args.command == "generate":
        from cli.cmd_generate import cmd_generate
        return cmd_generate(args)

    elif args.command == "serve":
        from cli.cmd_generate import cmd_serve
        return cmd_serve(args)

    elif args.command == "ablation":
        from cli.cmd_ablation import cmd_ablation
        return cmd_ablation(args)

    elif args.command == "dashboard":
        from cli.cmd_dashboard import cmd_dashboard
        return cmd_dashboard(args)

    elif args.command == "cloud":
        if not args.cloud_sub:
            # Default to status
            args.cloud_sub = "status"
        from cli.cmd_cloud import cmd_cloud
        return cmd_cloud(args)

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
