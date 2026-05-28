"""Load routing config + provider credentials."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_ROUTING_YAML_CANDIDATES = [
    Path("/root/llm-gateway/config/routing.yaml"),
    Path(__file__).parent.parent.parent / "config" / "routing.yaml",
]
_ROUTING_YAML = next((p for p in _ROUTING_YAML_CANDIDATES if p.exists()), _ROUTING_YAML_CANDIDATES[0])
_GLOBAL_ENV = Path("/root/.env")

_config: dict[str, Any] | None = None


def _load_env() -> None:
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
    base_url: str = pcfg.get("base_url") or os.environ.get(pcfg.get("base_url_env", ""), "")
    api_key: str = os.environ.get(pcfg["api_key_env"], "")
    return base_url.rstrip("/"), api_key
