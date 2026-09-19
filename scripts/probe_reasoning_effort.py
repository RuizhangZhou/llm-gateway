#!/usr/bin/env python3
"""Probe KI Connect reasoning-effort support with tiny live requests.

The provider's /models endpoint only tells us that a model exists. This tool
records the per-model request values actually accepted by /chat/completions,
then writes the result into routing.yaml for the gateway client to use.

Usage:
    uv run python scripts/probe_reasoning_effort.py --write
    uv run python scripts/probe_reasoning_effort.py --write --models gpt-5.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROUTING_YAML = Path(os.getenv("LLM_GATEWAY_ROUTING_FILE", PROJECT_ROOT / "config/routing.yaml"))
ENV_FILE = Path(os.getenv("LLM_GATEWAY_ENV_FILE", "/root/.env"))
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")


def _load_env() -> None:
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() and key.strip() not in os.environ:
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


def _is_gpt5(model: str) -> bool:
    lower = model.lower()
    return lower.startswith("gpt-5") and "gpt-oss" not in lower


def _target_map(model: str, supported: list[str]) -> dict[str, str]:
    """Map task intent to a live-supported effort without wasting quota."""
    lower = model.lower()
    if "mistral" in lower:
        preferred = {"none": "none", "low": "none", "medium": "high", "high": "high", "xhigh": "high"}
    elif "qwen" in lower:
        preferred = {"none": "none", "low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh"}
    elif "gpt-oss" in lower:
        preferred = {"none": "low", "low": "low", "medium": "medium", "high": "high", "xhigh": "high"}
    else:
        preferred = {name: name for name in EFFORTS}
    return {target: actual for target, actual in preferred.items() if actual in supported}


def _probe(base_url: str, api_key: str, model: str, effort: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "reasoning_effort": effort,
    }
    body["max_completion_tokens" if _is_gpt5(model) else "max_tokens"] = 32
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=75) as response:
            payload = json.loads(response.read())
        choice = (payload.get("choices") or [{}])[0]
        return {
            "effort": effort,
            "accepted": True,
            "status": response.status,
            "latency_ms": round((time.monotonic() - started) * 1000),
            "finish_reason": choice.get("finish_reason"),
        }
    except urllib.error.HTTPError as error:
        # Error bodies are retained only as a small diagnostic; never include
        # headers or credentials in terminal/config output.
        return {
            "effort": effort,
            "accepted": False,
            "status": error.code,
            "error": error.read().decode("utf-8", "replace")[:300].replace("\n", " "),
        }
    except Exception as error:
        return {"effort": effort, "accepted": False, "status": "error", "error": type(error).__name__}


def _atomic_write(path: Path, config: dict[str, Any]) -> None:
    rendered = yaml.safe_dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(rendered)
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the live results to routing.yaml")
    parser.add_argument("--models", nargs="+", help="model IDs to probe; defaults to all KI text models in routing.yaml")
    args = parser.parse_args()

    _load_env()
    config = yaml.safe_load(ROUTING_YAML.read_text(encoding="utf-8"))
    provider = config["providers"]["kiconnect"]
    base_url = provider["base_url"]
    api_key = os.environ.get(provider["api_key_env"], "")
    if not api_key:
        raise RuntimeError(f"Missing {provider['api_key_env']}; not probing")

    catalog = config.get("model_catalog", {})
    models = args.models or [key.removeprefix("kiconnect:") for key in catalog if key.startswith("kiconnect:")]
    if not models:
        raise RuntimeError("No KI Connect text models are present in routing.yaml")

    probed_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    results: dict[str, list[dict[str, Any]]] = {}
    for model in models:
        attempts = [_probe(base_url, api_key, model, effort) for effort in EFFORTS]
        results[model] = attempts
        accepted = [item["effort"] for item in attempts if item["accepted"]]
        print(f"{model}: accepted={','.join(accepted) or '(none)'}")

    if args.write:
        catalog = config.setdefault("model_catalog", {})
        for model, attempts in results.items():
            accepted = [item["effort"] for item in attempts if item["accepted"]]
            key = f"kiconnect:{model}"
            entry = catalog.setdefault(key, {"available": True})
            entry["reasoning_effort"] = {
                "supported": accepted,
                "task_target_map": _target_map(model, accepted),
                "probed_at": probed_at,
                "source": "live_http_probe",
                "request_limit_tokens": 32,
            }
        config.setdefault("metadata", {})["reasoning_effort_probed_at"] = probed_at
        _atomic_write(ROUTING_YAML, config)
        print(f"Wrote {ROUTING_YAML}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"reasoning effort probe failed: {error}", file=sys.stderr)
        raise SystemExit(1)
