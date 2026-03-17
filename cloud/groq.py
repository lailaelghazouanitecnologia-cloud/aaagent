"""Groq API health check — validates API key and model availability.

The actual Groq integration lives in eval/llm_judge.py. This module
provides quick connectivity checks used by `z86 doctor`.
"""

from __future__ import annotations

import os


def check_api_key() -> tuple[bool, str]:
    """Validate GROQ_API_KEY by listing models via the Groq SDK.

    Returns:
        (success, detail) — detail is model count on success, error message on failure.
    """
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        return False, "GROQ_API_KEY not set"

    try:
        from groq import Groq

        client = Groq(api_key=key, timeout=10)
        models = client.models.list()
        count = len(models.data)
        return True, f"{count} models available"
    except Exception as e:
        return False, f"Connection error: {e}"
