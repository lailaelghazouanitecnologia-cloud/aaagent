"""
Lightweight reporter that sends training metrics to the HCLM-D dashboard.

Usage in training loop:
    from training.dashboard_reporter import DashboardReporter

    reporter = DashboardReporter(run="A2-clusters-only")

    for step in range(total_steps):
        # ... training step ...

        reporter.report(
            step=step,
            losses={"total": loss.item(), "diffusion": l_diff.item(), ...},
            cluster_health={"entropy_ratio": 0.87, "dead_clusters": 0},
            gate={"mean": gate.mean().item(), "std": gate.std().item()},
            throughput={"tokens_per_sec": tps, "gpu_memory_gb": mem},
        )

    reporter.close()
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import deque
from typing import Any

import requests

logger = logging.getLogger(__name__)

DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "http://localhost:3000")


class DashboardReporter:
    """Async-safe reporter that batches and sends metrics to the dashboard API."""

    def __init__(
        self,
        run: str,
        url: str = DASHBOARD_URL,
        batch_size: int = 10,
        log_interval: int = 100,
    ):
        self.run = run
        self.url = url.rstrip("/")
        self.batch_size = batch_size
        self.log_interval = log_interval
        self._buffer: deque[dict[str, Any]] = deque()
        self._lock = threading.Lock()
        self._enabled = True
        self._step_count = 0

        # Test connectivity
        try:
            r = requests.get(f"{self.url}/api/health", timeout=3)
            if r.ok:
                logger.info(f"Dashboard connected at {self.url}")
            else:
                logger.warning(f"Dashboard returned {r.status_code}, disabling reporter")
                self._enabled = False
        except requests.ConnectionError:
            logger.warning(f"Dashboard not reachable at {self.url}, disabling reporter")
            self._enabled = False

    def report(
        self,
        step: int,
        losses: dict[str, float] | None = None,
        cluster_health: dict[str, Any] | None = None,
        gate: dict[str, float] | None = None,
        hierarchy: dict[str, float] | None = None,
        throughput: dict[str, float] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Queue a metric datapoint. Flushes when batch_size is reached."""
        if not self._enabled:
            return

        payload: dict[str, Any] = {"run": self.run, "step": step}
        if losses:
            payload["losses"] = losses
        if cluster_health:
            payload["cluster_health"] = cluster_health
        if gate:
            payload["gate"] = gate
        if hierarchy:
            payload["hierarchy"] = hierarchy
        if throughput:
            payload["throughput"] = throughput
        if extra:
            payload["extra"] = extra

        with self._lock:
            self._buffer.append(payload)
            self._step_count += 1

            if len(self._buffer) >= self.batch_size:
                self._flush()

    def report_generation(
        self,
        step: int,
        text: str,
        prompt: str | None = None,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Report a generated sample for the Generation page."""
        extra: dict[str, Any] = {"generated_sample": text}
        if prompt:
            extra["prompt"] = prompt
        if metrics:
            extra["sample_metrics"] = metrics
        self.report(step=step, extra=extra)

    def _flush(self) -> None:
        """Send buffered metrics to the dashboard. Called with lock held."""
        if not self._buffer:
            return

        batch = list(self._buffer)
        self._buffer.clear()

        # Fire and forget in background thread
        thread = threading.Thread(target=self._send, args=(batch,), daemon=True)
        thread.start()

    def _send(self, batch: list[dict]) -> None:
        try:
            if len(batch) == 1:
                requests.post(
                    f"{self.url}/api/metrics",
                    json=batch[0],
                    timeout=5,
                )
            else:
                requests.post(
                    f"{self.url}/api/metrics/batch",
                    json=batch,
                    timeout=10,
                )
        except Exception as e:
            logger.debug(f"Failed to send metrics to dashboard: {e}")

    def flush(self) -> None:
        """Manually flush any buffered metrics."""
        with self._lock:
            self._flush()

    def close(self) -> None:
        """Flush remaining metrics and disable the reporter."""
        self.flush()
        self._enabled = False
