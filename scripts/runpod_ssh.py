"""RunPod SSH helper — upload project and execute commands via paramiko.

Usage:
  # Execute a command on the pod
  python scripts/runpod_ssh.py exec \\
    --host ssh.runpod.io --user USER --key ~/.ssh/id_ed25519 \\
    -- "nvidia-smi"

  # Upload project to the pod
  python scripts/runpod_ssh.py upload \\
    --host ssh.runpod.io --user USER --key ~/.ssh/id_ed25519 \\
    --local-dir . --remote-dir /workspace/aaagent
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def get_ssh_client(host: str, user: str, key_path: str, port: int = 22):
    """Create and return a connected paramiko SSH client."""
    import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username=user, key_filename=key_path)
    return client


def cmd_exec(args):
    """Execute a command on the remote pod."""
    client = get_ssh_client(args.host, args.user, args.key, args.port)
    command = " ".join(args.command)
    print(f"→ Executing: {command}")

    stdin, stdout, stderr = client.exec_command(command)
    exit_code = stdout.channel.recv_exit_status()

    out = stdout.read().decode()
    err = stderr.read().decode()

    if out:
        print(out)
    if err:
        print(err, file=sys.stderr)

    client.close()
    return exit_code


def cmd_upload(args):
    """Upload local directory to remote pod via SFTP."""
    import paramiko

    client = get_ssh_client(args.host, args.user, args.key, args.port)
    sftp = client.open_sftp()

    local_dir = Path(args.local_dir).resolve()
    remote_dir = args.remote_dir

    # Patterns to skip
    skip = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist",
            ".mypy_cache", ".ruff_cache", ".pytest_cache", "*.pyc", ".env"}

    def should_skip(name: str) -> bool:
        return name in skip or name.startswith(".")

    def _mkdir_p(remote_path: str):
        """Recursively create remote directories."""
        dirs_to_create = []
        current = remote_path
        while current and current != "/":
            try:
                sftp.stat(current)
                break
            except FileNotFoundError:
                dirs_to_create.append(current)
                current = os.path.dirname(current)
        for d in reversed(dirs_to_create):
            sftp.mkdir(d)

    count = 0
    for root, dirs, files in os.walk(local_dir):
        dirs[:] = [d for d in dirs if not should_skip(d)]
        rel = os.path.relpath(root, local_dir)
        remote_subdir = os.path.join(remote_dir, rel) if rel != "." else remote_dir

        _mkdir_p(remote_subdir)

        for fname in files:
            if should_skip(fname):
                continue
            local_path = os.path.join(root, fname)
            remote_path = os.path.join(remote_subdir, fname)
            sftp.put(local_path, remote_path)
            count += 1

    print(f"✓ Uploaded {count} files to {remote_dir}")
    sftp.close()
    client.close()


def main():
    parser = argparse.ArgumentParser(description="RunPod SSH helper")
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--key", required=True, help="Path to SSH private key")
    parser.add_argument("--port", type=int, default=22)

    sub = parser.add_subparsers(dest="action", required=True)

    exec_p = sub.add_parser("exec", help="Execute command on pod")
    exec_p.add_argument("command", nargs="+")

    upload_p = sub.add_parser("upload", help="Upload directory to pod")
    upload_p.add_argument("--local-dir", default=".", help="Local directory to upload")
    upload_p.add_argument("--remote-dir", default="/workspace/aaagent", help="Remote destination")

    args = parser.parse_args()

    if args.action == "exec":
        sys.exit(cmd_exec(args))
    elif args.action == "upload":
        cmd_upload(args)


if __name__ == "__main__":
    main()
