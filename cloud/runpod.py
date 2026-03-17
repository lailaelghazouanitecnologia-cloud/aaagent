"""RunPod GPU cloud management — start/stop/status pods for training & inference.

Env vars:
    RUNPOD_API_KEY  — RunPod API key (required)

Usage:
    from cloud.runpod import RunPodClient
    client = RunPodClient()
    pod = client.create_pod("hclm-d-train", gpu_type="NVIDIA RTX A6000")
    client.status(pod["id"])
    client.stop(pod["id"])
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger(__name__)

RUNPOD_API_URL = "https://api.runpod.io/graphql"

# GPU presets for common training/inference scenarios
GPU_PRESETS = {
    "train-small": {
        "gpu_type": "NVIDIA RTX A6000",
        "gpu_count": 1,
        "volume_gb": 50,
        "container_image": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
        "description": "Single A6000 — quick training runs (~24h for 100k steps)",
    },
    "train-fast": {
        "gpu_type": "NVIDIA A100 80GB PCIe",
        "gpu_count": 1,
        "volume_gb": 100,
        "container_image": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
        "description": "Single A100 — full training in ~12h",
    },
    "inference": {
        "gpu_type": "NVIDIA RTX 4090",
        "gpu_count": 1,
        "volume_gb": 20,
        "container_image": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
        "description": "RTX 4090 — low-cost inference serving",
    },
}


@dataclass
class PodInfo:
    """Parsed pod metadata."""
    id: str
    name: str
    status: str
    gpu_type: str = ""
    gpu_count: int = 0
    cost_per_hr: float = 0.0
    uptime_hrs: float = 0.0
    ssh_command: str = ""
    ports: dict[str, str] = field(default_factory=dict)


class RunPodClient:
    """Thin wrapper around RunPod GraphQL API."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY", "")
        if not self.api_key:
            raise ValueError("RUNPOD_API_KEY not set")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _query(self, query: str, variables: dict | None = None) -> dict[str, Any]:
        """Execute a GraphQL query with retry."""
        payload: dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables

        for attempt in range(4):
            try:
                resp = requests.post(
                    RUNPOD_API_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                if "errors" in data:
                    raise RuntimeError(f"RunPod API error: {data['errors']}")
                return data.get("data", {})
            except requests.RequestException as e:
                if attempt < 3:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"RunPod request failed (attempt {attempt+1}), retry in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise

        return {}  # unreachable, but satisfies type checker

    # ── Pod lifecycle ──

    def list_pods(self) -> list[PodInfo]:
        """List all active pods."""
        query = """
        query { myself { pods {
            id name desiredStatus
            machine { gpuDisplayName }
            gpuCount costPerHr uptimeSeconds
            runtime { ports { ip isIpPublic privatePort publicPort type } }
        }}}
        """
        data = self._query(query)
        pods = data.get("myself", {}).get("pods", [])
        results = []
        for p in pods:
            ports = {}
            if p.get("runtime") and p["runtime"].get("ports"):
                for port in p["runtime"]["ports"]:
                    if port.get("isIpPublic"):
                        ports[str(port["privatePort"])] = f"{port['ip']}:{port['publicPort']}"

            ssh_cmd = ""
            if "22" in ports:
                ssh_cmd = f"ssh root@{ports['22'].split(':')[0]} -p {ports['22'].split(':')[1]}"

            results.append(PodInfo(
                id=p["id"],
                name=p.get("name", ""),
                status=p.get("desiredStatus", "unknown"),
                gpu_type=p.get("machine", {}).get("gpuDisplayName", ""),
                gpu_count=p.get("gpuCount", 0),
                cost_per_hr=p.get("costPerHr", 0),
                uptime_hrs=p.get("uptimeSeconds", 0) / 3600,
                ssh_command=ssh_cmd,
                ports=ports,
            ))
        return results

    def list_volumes(self) -> list[dict[str, Any]]:
        """List all network volumes."""
        query = """
        query { myself { networkVolumes {
            id name size dataCenterId
        }}}
        """
        data = self._query(query)
        return data.get("myself", {}).get("networkVolumes", [])

    def create_pod(
        self,
        name: str = "hclm-d",
        preset: str | None = None,
        gpu_type: str | None = None,
        gpu_count: int = 1,
        volume_gb: int = 50,
        volume_id: str | None = None,
        container_image: str | None = None,
        docker_args: str = "",
        ports: str = "22/tcp,8080/http,3000/http",
    ) -> PodInfo:
        """Create a new GPU pod.

        Args:
            volume_id: Existing network volume ID to mount at /workspace.
                       If set, volume_gb is ignored and the existing volume is used.
        """
        if preset and preset in GPU_PRESETS:
            cfg = GPU_PRESETS[preset]
            gpu_type = gpu_type or cfg["gpu_type"]
            gpu_count = cfg.get("gpu_count", gpu_count)
            volume_gb = cfg.get("volume_gb", volume_gb)
            container_image = container_image or cfg["container_image"]

        gpu_type = gpu_type or "NVIDIA RTX A6000"
        container_image = container_image or "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"

        query = """
        mutation($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id name desiredStatus
                machine { gpuDisplayName }
                gpuCount costPerHr
            }
        }
        """
        pod_input: dict[str, Any] = {
            "name": name,
            "imageName": container_image,
            "gpuTypeId": gpu_type,
            "gpuCount": gpu_count,
            "containerDiskInGb": 20,
            "ports": ports,
            "dockerArgs": docker_args,
        }

        if volume_id:
            pod_input["networkVolumeId"] = volume_id
        else:
            pod_input["volumeInGb"] = volume_gb

        data = self._query(query, {"input": pod_input})
        pod = data.get("podFindAndDeployOnDemand", {})
        return PodInfo(
            id=pod.get("id", ""),
            name=pod.get("name", name),
            status=pod.get("desiredStatus", "CREATED"),
            gpu_type=pod.get("machine", {}).get("gpuDisplayName", gpu_type),
            gpu_count=pod.get("gpuCount", gpu_count),
            cost_per_hr=pod.get("costPerHr", 0),
        )

    def stop(self, pod_id: str) -> bool:
        """Stop a running pod."""
        query = """
        mutation($input: PodStopInput!) {
            podStop(input: $input) { id desiredStatus }
        }
        """
        data = self._query(query, {"input": {"podId": pod_id}})
        return data.get("podStop", {}).get("desiredStatus") == "EXITED"

    def start(self, pod_id: str) -> bool:
        """Resume a stopped pod."""
        query = """
        mutation($input: PodResumeInput!) {
            podResume(input: $input) { id desiredStatus }
        }
        """
        data = self._query(query, {"input": {"podId": pod_id}})
        return data.get("podResume", {}).get("desiredStatus") == "RUNNING"

    def terminate(self, pod_id: str) -> bool:
        """Permanently destroy a pod (irreversible)."""
        query = """
        mutation($input: PodTerminateInput!) {
            podTerminate(input: $input)
        }
        """
        self._query(query, {"input": {"podId": pod_id}})
        return True

    def get_gpu_types(self) -> list[dict[str, Any]]:
        """List available GPU types and pricing."""
        query = """
        query { gpuTypes {
            id displayName memoryInGb
            secureCloud lowestPrice { minimumBidPrice uninterruptablePrice }
        }}
        """
        data = self._query(query)
        return data.get("gpuTypes", [])


def check_api_key() -> bool:
    """Quick validation that RUNPOD_API_KEY is set and works."""
    key = os.environ.get("RUNPOD_API_KEY", "")
    if not key:
        return False
    try:
        client = RunPodClient(key)
        client.list_pods()
        return True
    except Exception:
        return False
