"""Load routing config + provider credentials."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Canonical config location on this server; also check package source as fallback
_ROUTING_YAML_CANDIDATES = [
    Path("/root/llm-gateway/config/routing.yaml"),
    Path(__file__).parent.parent.parent / "config" / "routing.yaml",
]
_ROUTING_YAML = next((p for p in _ROUTING_YAML_CANDIDATES if p.exists()), _ROUTING_YAML_CANDIDATES[0])
_KICONNECT_QUOTA = Path("/root/.openclaw/kiconnect/quota.json")
_GLOBAL_ENV = Path("/root/.env")

_config: dict[str, Any] | None = None


def _load_env() -> None:
    """Load /root/.env into os.environ if not already present (non-override)."""
    if not _GLOBAL_ENV.exists():
        return
    for line in _GLOBAL_ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def get_config() -> dict[str, Any]:
    global _config
    if _config is not None:
        return _config
    _load_env()
    _config = yaml.safe_load(_ROUTING_YAML.read_text(encoding="utf-8"))
    return _config


def provider_credentials(provider: str) -> tuple[str, str]:
    """Return (base_url, api_key) for a provider name."""
    cfg = get_config()
    pcfg = cfg["providers"][provider]

    base_url: str = pcfg.get("base_url") or os.environ.get(pcfg["base_url_env"], "")
    api_key: str = os.environ.get(pcfg["api_key_env"], "")
    return base_url.rstrip("/"), api_key


def kiconnect_available_models() -> list[str]:
    """Return text-generation KIconnect model IDs that are currently available.

    Reads /root/.openclaw/kiconnect/quota.json (maintained by kiconnect-refresh.timer).
    Falls back to hardcoded list if file doesn't exist.
    """
    import json

    if _KICONNECT_QUOTA.exists():
        try:
            data = json.loads(_KICONNECT_QUOTA.read_text(encoding="utf-8"))
            models = data.get("models", {})
            available = [
                mid for mid, meta in models.items()
                if isinstance(meta, dict)
                and meta.get("availableNow", True)
                and meta.get("category", "chat") == "chat"
                and meta.get("useByDefault", True) is not False
            ]
            if available:
                return available
        except Exception:
            pass

    # Fallback: known models as of last refresh
    return ["gpt-oss-120b", "gpt-5.2", "gpt-5.4-mini", "mistral-small-4-119b-2603"]
