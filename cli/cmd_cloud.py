"""z86 cloud — manage RunPod GPU pods and Cloudflare DNS."""

from __future__ import annotations

import os
import sys

from cli import ui


def cmd_cloud(args):
    """Cloud infrastructure management."""
    ui.logo()

    sub = args.cloud_sub
    if sub == "status":
        return _cloud_status()
    elif sub == "gpus":
        return _list_gpus()
    elif sub == "start":
        return _start_pod(args)
    elif sub == "stop":
        return _stop_pod(args)
    elif sub == "terminate":
        return _terminate_pod(args)
    elif sub == "ssh":
        return _ssh_pod(args)
    elif sub == "dns":
        return _dns_status()
    elif sub == "dns-set":
        return _dns_set(args)
    else:
        ui.err(f"Unknown subcommand: {sub}")
        return 1


# ── Status ──

def _cloud_status():
    """Show all cloud resources status."""
    ui.step("Cloud Status")

    # Groq
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        try:
            from cloud.groq import check_api_key
            ok, detail = check_api_key()
            ui.ok(f"Groq: {detail}") if ok else ui.warn(f"Groq: {detail}")
        except Exception as e:
            ui.warn(f"Groq: {e}")
    else:
        ui.info("Groq: not configured")

    # RunPod
    runpod_key = os.environ.get("RUNPOD_API_KEY", "")
    if runpod_key:
        try:
            from cloud.runpod import RunPodClient
            client = RunPodClient(runpod_key)
            pods = client.list_pods()
            if pods:
                ui.step(f"RunPod Pods ({len(pods)})")
                headers = ["ID", "Name", "Status", "GPU", "$/hr", "Uptime"]
                rows = []
                for p in pods:
                    status_color = ui.C.GREEN if p.status == "RUNNING" else ui.C.YELLOW
                    rows.append([
                        f"{ui.C.WHITE}{p.id}{ui.C.RST}",
                        p.name,
                        f"{status_color}{p.status}{ui.C.RST}",
                        f"{p.gpu_type} x{p.gpu_count}",
                        f"${p.cost_per_hr:.2f}",
                        f"{p.uptime_hrs:.1f}h",
                    ])
                ui.table(headers, rows)
            else:
                ui.info("RunPod: no active pods")
        except Exception as e:
            ui.warn(f"RunPod: {e}")
    else:
        ui.info("RunPod: not configured")

    # Cloudflare
    cf_token = os.environ.get("CF_API_TOKEN", "")
    cf_zone = os.environ.get("CF_ZONE_ID", "")
    if cf_token and cf_zone:
        try:
            from cloud.cloudflare import CloudflareClient
            cf = CloudflareClient(cf_token, cf_zone)
            records = cf.list_dns()
            a_records = [r for r in records if r.type in ("A", "AAAA", "CNAME")]
            if a_records:
                ui.step(f"Cloudflare DNS ({len(a_records)} records)")
                headers = ["Type", "Name", "Content", "Proxied"]
                rows = []
                for r in a_records:
                    proxy_str = f"{ui.C.GREEN}yes{ui.C.RST}" if r.proxied else "no"
                    rows.append([r.type, r.name, r.content, proxy_str])
                ui.table(headers, rows)
            else:
                ui.info("Cloudflare: no A/AAAA/CNAME records")
        except Exception as e:
            ui.warn(f"Cloudflare: {e}")
    else:
        ui.info("Cloudflare: not configured (set CF_API_TOKEN + CF_ZONE_ID)")

    return 0


# ── RunPod GPU listing ──

def _list_gpus():
    """List available GPU types on RunPod."""
    from cloud.runpod import RunPodClient, GPU_PRESETS

    ui.step("GPU Presets")
    for name, cfg in GPU_PRESETS.items():
        ui.kv(name, cfg["description"])

    try:
        client = RunPodClient()
        gpus = client.get_gpu_types()
        ui.step(f"Available on RunPod ({len(gpus)} types)")
        headers = ["GPU", "VRAM", "Spot $/hr", "On-demand $/hr"]
        rows = []
        for g in sorted(gpus, key=lambda x: x.get("memoryInGb", 0)):
            if not g.get("secureCloud"):
                continue
            price = g.get("lowestPrice", {})
            rows.append([
                g.get("displayName", g.get("id", "")),
                f"{g.get('memoryInGb', '?')}GB",
                f"${price.get('minimumBidPrice', '?'):.3f}" if price.get("minimumBidPrice") else "—",
                f"${price.get('uninterruptablePrice', '?'):.3f}" if price.get("uninterruptablePrice") else "—",
            ])
        ui.table(headers, rows[:20])  # Cap display
        if len(rows) > 20:
            ui.info(f"... and {len(rows) - 20} more")
    except Exception as e:
        ui.err(f"Failed to list GPUs: {e}")
        return 1
    return 0


# ── Pod lifecycle ──

def _start_pod(args):
    """Create and start a new GPU pod."""
    from cloud.runpod import RunPodClient

    preset = getattr(args, "preset", None)
    name = getattr(args, "pod_name", "hclm-d")

    ui.step(f"Creating pod '{name}'" + (f" (preset: {preset})" if preset else ""))

    try:
        client = RunPodClient()
        pod = client.create_pod(name=name, preset=preset)
        ui.ok(f"Pod created: {pod.id}")
        ui.kv("GPU", f"{pod.gpu_type} x{pod.gpu_count}")
        ui.kv("Cost", f"${pod.cost_per_hr:.2f}/hr")
        ui.kv("Status", pod.status)
        ui.info("")
        ui.info("Wait for RUNNING status, then:")
        ui.info(f"  z86 cloud ssh {pod.id}")
        ui.info(f"  z86 cloud stop {pod.id}")
    except Exception as e:
        ui.err(f"Failed: {e}")
        return 1
    return 0


def _stop_pod(args):
    """Stop a running pod."""
    from cloud.runpod import RunPodClient

    pod_id = args.pod_id
    ui.step(f"Stopping pod {pod_id}")
    try:
        client = RunPodClient()
        if client.stop(pod_id):
            ui.ok("Pod stopped (billing paused)")
            ui.info(f"Resume with: z86 cloud start {pod_id}")
        else:
            ui.warn("Stop command sent but status unclear")
    except Exception as e:
        ui.err(f"Failed: {e}")
        return 1
    return 0


def _terminate_pod(args):
    """Permanently destroy a pod."""
    from cloud.runpod import RunPodClient

    pod_id = args.pod_id
    if not ui.confirm(f"Permanently destroy pod {pod_id}? This deletes all data on the pod."):
        ui.abort()
        return 1

    try:
        client = RunPodClient()
        client.terminate(pod_id)
        ui.ok(f"Pod {pod_id} terminated")
    except Exception as e:
        ui.err(f"Failed: {e}")
        return 1
    return 0


def _ssh_pod(args):
    """Print SSH command for a pod."""
    from cloud.runpod import RunPodClient

    pod_id = args.pod_id
    try:
        client = RunPodClient()
        pods = client.list_pods()
        pod = next((p for p in pods if p.id == pod_id), None)
        if not pod:
            ui.err(f"Pod {pod_id} not found")
            return 1
        if pod.ssh_command:
            ui.ok(f"SSH: {pod.ssh_command}")
            ui.info("Ports available:")
            for private, public in pod.ports.items():
                ui.kv(f"  :{private}", public)
        else:
            ui.warn("Pod not ready or SSH port not exposed")
    except Exception as e:
        ui.err(f"Failed: {e}")
        return 1
    return 0


# ── Cloudflare DNS ──

def _dns_status():
    """Show DNS records."""
    from cloud.cloudflare import CloudflareClient

    try:
        cf = CloudflareClient()
        records = cf.list_dns()
        ui.step(f"DNS Records ({len(records)})")
        headers = ["Type", "Name", "Content", "Proxied", "TTL"]
        rows = []
        for r in records:
            proxy_str = f"{ui.C.GREEN}proxied{ui.C.RST}" if r.proxied else "direct"
            ttl_str = "auto" if r.ttl == 1 else f"{r.ttl}s"
            rows.append([r.type, r.name, r.content, proxy_str, ttl_str])
        ui.table(headers, rows)
    except Exception as e:
        ui.err(f"Failed: {e}")
        return 1
    return 0


def _dns_set(args):
    """Create or update a DNS record."""
    from cloud.cloudflare import CloudflareClient

    name = args.dns_name
    record_type = getattr(args, "dns_type", "A")
    content = args.dns_content
    proxied = not getattr(args, "no_proxy", False)

    ui.step(f"Setting {record_type} record: {name} → {content}")
    try:
        cf = CloudflareClient()
        record = cf.upsert_dns(name, record_type, content, proxied=proxied)
        ui.ok(f"DNS record set: {record.name} → {record.content}")
        ui.kv("Proxied", "yes" if record.proxied else "no")
    except Exception as e:
        ui.err(f"Failed: {e}")
        return 1
    return 0
