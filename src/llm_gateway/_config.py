"""Load routing config + provider credentials."""

from __future__ import annotations

import os
import logging
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml

_ROUTING_YAML_CANDIDATES = [
    Path("/root/llm-gateway/config/routing.yaml"),
    Path(__file__).parent.parent.parent / "config" / "routing.yaml",
]
_ROUTING_YAML = Path(
    os.getenv(
        "LLM_GATEWAY_ROUTING_FILE",
        str(next((p for p in _ROUTING_YAML_CANDIDATES if p.exists()), _ROUTING_YAML_CANDIDATES[0])),
    )
)
_GLOBAL_ENV = Path(os.getenv("LLM_GATEWAY_ENV_FILE", "/root/.env"))
_CONFIG_REPO = os.getenv("LLM_GATEWAY_CONFIG_REPO", "RuizhangZhou/llm-gateway")
_CONFIG_REF = os.getenv("LLM_GATEWAY_CONFIG_REF", "master")
_CONFIG_PATH = os.getenv("LLM_GATEWAY_CONFIG_PATH", "config/routing.yaml")
_CONFIG_URL = os.getenv(
    "LLM_GATEWAY_CONFIG_URL",
    f"https://raw.githubusercontent.com/{_CONFIG_REPO}/{quote(_CONFIG_REF, safe='')}/{quote(_CONFIG_PATH, safe='/')}",
)
_CONFIG_REFRESH_SECONDS = max(0, int(os.getenv("LLM_GATEWAY_CONFIG_REFRESH_SECONDS", "300")))
_CACHE_ROOT = Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache")))
_CONFIG_CACHE = Path(
    os.getenv("LLM_GATEWAY_ROUTING_CACHE", str(_CACHE_ROOT / "llm-gateway" / "routing.yaml"))
)

_config: dict[str, Any] | None = None
_last_remote_check = 0.0
_logger = logging.getLogger("llm_gateway.config")


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


def _read_yaml(path: Path) -> dict[str, Any] | None:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        return None
    return value if isinstance(value, dict) else None


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(content)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _fetch_shared_config() -> dict[str, Any] | None:
    request = urllib.request.Request(_CONFIG_URL, headers={"Accept": "text/plain"})
    token = (
        os.environ.get("LLM_GATEWAY_GITHUB_TOKEN")
        or os.environ.get("GH_TOKEN")
        or os.environ.get("GITHUB_TOKEN")
    )
    if token and _CONFIG_URL.startswith("https://raw.githubusercontent.com/"):
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            content = response.read().decode("utf-8")
        value = yaml.safe_load(content)
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("providers"), dict)
            or not isinstance(value.get("task_routing"), dict)
        ):
            raise ValueError("shared routing config has an invalid structure")
        # Cache only validated configuration, preserving the last known-good
        # copy if GitHub is unavailable or the published file is malformed.
        try:
            _atomic_write(_CONFIG_CACHE, content)
        except OSError as exc:
            _logger.debug("Could not persist routing config cache: %s", exc)
        return value
    except (OSError, UnicodeError, yaml.YAMLError, ValueError, urllib.error.URLError) as exc:
        _logger.debug("Could not refresh shared routing config: %s", exc)
        return None


def _local_config() -> dict[str, Any] | None:
    # The cache is updated only after a successful GitHub fetch. On a fresh
    # machine without a cache, use the packaged/check-out config as bootstrap.
    return _read_yaml(_CONFIG_CACHE) or _read_yaml(_ROUTING_YAML)


def get_config() -> dict[str, Any]:
    global _config, _last_remote_check
    _load_env()
    now = time.monotonic()
    should_refresh = _config is None or (
        _CONFIG_REFRESH_SECONDS > 0 and now - _last_remote_check >= _CONFIG_REFRESH_SECONDS
    )
    if should_refresh:
        _last_remote_check = now
        latest = _fetch_shared_config()
        if latest is not None:
            _config = latest
        elif _config is None:
            _config = _local_config()
    if _config is None:
        raise ValueError(f"No valid shared or local routing config; checked {_CONFIG_URL} and {_ROUTING_YAML}")
    return _config


def provider_credentials(provider: str) -> tuple[str, str]:
    """Return (base_url, api_key) for a provider name."""
    cfg = get_config()
    pcfg = cfg["providers"][provider]
    base_url: str = pcfg.get("base_url") or os.environ.get(pcfg.get("base_url_env", ""), "")
    api_key: str = os.environ.get(pcfg["api_key_env"], "")
    return base_url.rstrip("/"), api_key
