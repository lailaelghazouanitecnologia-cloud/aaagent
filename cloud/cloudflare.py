"""Cloudflare DNS management — programmatic DNS record creation for z86.dev.

Env vars:
    CF_API_TOKEN  — Cloudflare API token (DNS edit permissions)
    CF_ZONE_ID    — Zone ID for the domain

Usage:
    from cloud.cloudflare import CloudflareClient
    client = CloudflareClient()
    client.upsert_dns("z86.dev", "A", "1.2.3.4", proxied=True)
    client.list_dns()
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)

CF_API_BASE = "https://api.cloudflare.com/client/v4"


@dataclass
class DnsRecord:
    """A Cloudflare DNS record."""
    id: str
    type: str
    name: str
    content: str
    proxied: bool
    ttl: int


class CloudflareClient:
    """Thin wrapper around Cloudflare REST API for DNS management."""

    def __init__(self, api_token: str | None = None, zone_id: str | None = None):
        self.api_token = api_token or os.environ.get("CF_API_TOKEN", "")
        self.zone_id = zone_id or os.environ.get("CF_ZONE_ID", "")
        if not self.api_token:
            raise ValueError("CF_API_TOKEN not set")
        if not self.zone_id:
            raise ValueError("CF_ZONE_ID not set")
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        """Make API request with retry."""
        url = f"{CF_API_BASE}/zones/{self.zone_id}{path}"
        for attempt in range(4):
            try:
                resp = requests.request(method, url, headers=self.headers, timeout=30, **kwargs)
                resp.raise_for_status()
                data = resp.json()
                if not data.get("success"):
                    errors = data.get("errors", [])
                    raise RuntimeError(f"Cloudflare API error: {errors}")
                return data
            except requests.RequestException as e:
                if attempt < 3:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"Cloudflare request failed (attempt {attempt+1}), retry in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise
        return {}

    def list_dns(self, record_type: str | None = None) -> list[DnsRecord]:
        """List DNS records for the zone."""
        params: dict[str, Any] = {"per_page": 100}
        if record_type:
            params["type"] = record_type
        data = self._request("GET", "/dns_records", params=params)
        records = []
        for r in data.get("result", []):
            records.append(DnsRecord(
                id=r["id"],
                type=r["type"],
                name=r["name"],
                content=r["content"],
                proxied=r.get("proxied", False),
                ttl=r.get("ttl", 1),
            ))
        return records

    def upsert_dns(
        self,
        name: str,
        record_type: str = "A",
        content: str = "",
        proxied: bool = True,
        ttl: int = 1,
    ) -> DnsRecord:
        """Create or update a DNS record. TTL=1 means 'auto'."""
        # Check if record exists
        existing = [r for r in self.list_dns(record_type) if r.name == name and r.type == record_type]

        payload = {
            "type": record_type,
            "name": name,
            "content": content,
            "proxied": proxied,
            "ttl": ttl,
        }

        if existing:
            # Update
            record_id = existing[0].id
            data = self._request("PUT", f"/dns_records/{record_id}", json=payload)
        else:
            # Create
            data = self._request("POST", "/dns_records", json=payload)

        r = data.get("result", {})
        return DnsRecord(
            id=r.get("id", ""),
            type=r.get("type", record_type),
            name=r.get("name", name),
            content=r.get("content", content),
            proxied=r.get("proxied", proxied),
            ttl=r.get("ttl", ttl),
        )

    def delete_dns(self, record_id: str) -> bool:
        """Delete a DNS record."""
        self._request("DELETE", f"/dns_records/{record_id}")
        return True

    def verify_token(self) -> bool:
        """Verify the API token is valid."""
        try:
            url = f"{CF_API_BASE}/user/tokens/verify"
            resp = requests.get(url, headers=self.headers, timeout=10)
            data = resp.json()
            return data.get("success", False)
        except Exception:
            return False


def check_api_token() -> bool:
    """Quick validation that CF_API_TOKEN is set and works."""
    token = os.environ.get("CF_API_TOKEN", "")
    if not token:
        return False
    try:
        url = f"{CF_API_BASE}/user/tokens/verify"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        resp = requests.get(url, headers=headers, timeout=10)
        return resp.json().get("success", False)
    except Exception:
        return False
