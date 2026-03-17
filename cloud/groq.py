"""Groq API health check — validates API key and model availability.

The actual Groq integration lives in eval/llm_judge.py. This module
provides quick connectivity checks used by `z86 doctor`.
"""

from __future__ import annotations

import os

import requests


GROQ_API_URL = "https://api.groq.com/openai/v1"


def check_api_key() -> tuple[bool, str]:
    """Validate GROQ_API_KEY by listing models.

    Returns:
        (success, detail) — detail is model count on success, error message on failure.
    """
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        return False, "GROQ_API_KEY not set"

    try:
        resp = requests.get(
            f"{GROQ_API_URL}/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=10,
        )
        if resp.status_code == 401:
            return False, "Invalid API key"
        resp.raise_for_status()
        models = resp.json().get("data", [])
        return True, f"{len(models)} models available"
    except requests.RequestException as e:
        return False, f"Connection error: {e}"
