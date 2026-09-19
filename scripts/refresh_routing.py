#!/usr/bin/env python3
"""Discover live KIconnect models and atomically rebuild routing.yaml.

The model endpoint is authoritative for availability. Family rules below are
authoritative for cost tier and task fit, so a newly published version (for
example gpt-5.6) is placed correctly without waiting for an exact-name edit.

Usage:
    uv run python scripts/refresh_routing.py
    uv run python scripts/refresh_routing.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROUTING_YAML = Path(os.getenv("LLM_GATEWAY_ROUTING_FILE", PROJECT_ROOT / "config/routing.yaml"))
KICONNECT_BASE_URL = "https://chat.kiconnect.nrw/api/v1"
ENV_FILE = Path(os.getenv("LLM_GATEWAY_ENV_FILE", "/root/.env"))
GITHUB_REPO = os.getenv("LLM_GATEWAY_GITHUB_REPO", "RuizhangZhou/metaculus-bot")
GITHUB_ENVIRONMENT = os.getenv("LLM_GATEWAY_GITHUB_ENVIRONMENT", "metaculus bot")

EMBEDDING_HINTS = {"embedding", "e5-"}

# Provider diversity is valuable, but Azure is deliberately last: these jobs
# should consume KIconnect's free/quota-backed models before a paid fallback.
AZURE_MODELS: list[dict[str, Any]] = [
    {
        "id": "gpt-5.4",
        "tier": "frontier",
        "rank": 94,
        "capabilities": ["reasoning", "technical", "writing", "structured"],
    }
]


def _version_key(model_id: str) -> tuple[int, int, int]:
    match = re.search(r"gpt-(\d+)(?:\.(\d+))?", model_id.lower())
    if not match:
        return (0, 0, 0)
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    mini_penalty = -1 if "mini" in model_id.lower() else 0
    return (major, minor, mini_penalty)


def describe_model(model_id: str) -> dict[str, Any]:
    """Infer stable routing metadata from a model family/name."""
    lower = model_id.lower()
    if lower.startswith("gpt-5") and "gpt-oss" not in lower:
        version = _version_key(lower)
        is_mini = "mini" in lower
        return {
            "tier": "limited",
            "rank": 90 + min(version[1], 9) - (5 if is_mini else 0),
            "capabilities": ["reasoning", "technical", "forecast", "writing", "structured"],
            "family": "gpt-frontier",
        }
    if "qwen" in lower:
        return {
            "tier": "free",
            "rank": 82,
            "capabilities": ["multilingual", "agent", "structured", "technical"],
            "family": "qwen",
        }
    if "mistral" in lower:
        return {
            "tier": "free",
            "rank": 80,
            "capabilities": ["fast", "long_context", "writing", "structured", "technical"],
            "family": "mistral",
        }
    if "gpt-oss" in lower:
        return {
            "tier": "free",
            "rank": 78,
            "capabilities": ["reasoning", "agent", "coding", "structured"],
            "family": "gpt-oss",
        }
    return {
        "tier": "unknown",
        "rank": 10,
        "capabilities": [],
        "family": "unknown",
    }


def _load_env() -> None:
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def fetch_kiconnect_text_models(api_key: str) -> list[str]:
    import urllib.request

    req = urllib.request.Request(
        f"{KICONNECT_BASE_URL}/models",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        data = json.loads(response.read())

    results: list[str] = []
    for item in data.get("data", []):
        model_id = str(item.get("id", "")).strip()
        if not model_id or any(hint in model_id.lower() for hint in EMBEDDING_HINTS):
            continue
        results.append(model_id)
    if not results:
        raise RuntimeError("KIconnect returned no text-generation models; keeping last-known-good config")
    return list(dict.fromkeys(results))


def _ordered_by_family(models: list[str], families: tuple[str, ...]) -> list[str]:
    ordered: list[str] = []
    for family in families:
        matches = [model for model in models if describe_model(model)["family"] == family]
        if family == "gpt-frontier":
            matches.sort(key=_version_key, reverse=True)
        ordered.extend(matches)
    return ordered


def build_task_routing(model_ids: list[str]) -> dict[str, list[str]]:
    free = [model for model in model_ids if describe_model(model)["tier"] == "free"]
    limited = [model for model in model_ids if describe_model(model)["tier"] == "limited"]
    unknown = [model for model in model_ids if describe_model(model)["tier"] == "unknown"]

    frontier = _ordered_by_family(limited, ("gpt-frontier",))
    limited_economy = sorted(
        limited,
        key=lambda model: (0 if "mini" in model.lower() else 1, _version_key(model)),
    )
    writing = _ordered_by_family(free, ("mistral", "qwen", "gpt-oss"))
    multilingual = _ordered_by_family(free, ("qwen", "mistral", "gpt-oss"))
    reasoning = _ordered_by_family(free, ("gpt-oss", "mistral", "qwen"))
    azure = [f"azure:{entry['id']}" for entry in AZURE_MODELS]

    def chain(*groups: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for group in groups:
            for raw in group:
                item = raw if ":" in raw else f"kiconnect:{raw}"
                if item not in seen:
                    seen.add(item)
                    result.append(item)
        return result

    simple_writing = chain(writing, unknown, limited_economy, azure)
    simple_multilingual = chain(multilingual, unknown, limited_economy, azure)
    # Spend at most one scarce/frontier quota before returning to the free
    # pool. Older limited models remain as deep fallbacks, not a second paid
    # tier that gets consumed before free capacity.
    quality_first = chain(frontier[:1], reasoning, frontier[1:], unknown, azure)

    return {
        # Cheap/free-first production tasks.
        "social_post": list(simple_writing),
        "blog": list(simple_writing),
        "summarize": list(simple_writing),
        "youtube_digest": list(simple_multilingual),
        "classify": list(simple_multilingual),
        "job_score": list(simple_multilingual),
        "cold_email": list(simple_multilingual),
        "cv_tailor": list(simple_multilingual),
        "agent_step": list(simple_multilingual),
        # Quality-first tasks. Limited models fall through to all free models.
        "forecast": list(quality_first),
        "technical": list(quality_first),
        "code": list(quality_first),
        "agent_complex": list(quality_first),
        # Opt in only when an interactive caller values immediate output more
        # than deeper reasoning. Its task policy maps to effort=none.
        "instant": list(simple_writing),
        "default": list(simple_multilingual),
    }


def build_config(current: dict[str, Any], model_ids: list[str], now: str) -> dict[str, Any]:
    config = dict(current)
    config["metadata"] = {
        "refreshed_at": now,
        "source": f"{KICONNECT_BASE_URL}/models",
        "policy": "family rules in scripts/refresh_routing.py",
    }
    config["task_routing"] = build_task_routing(model_ids)

    previous_catalog = current.get("model_catalog", {})
    catalog: dict[str, dict[str, Any]] = {}
    for model_id in model_ids:
        key = f"kiconnect:{model_id}"
        catalog[key] = {
            **describe_model(model_id),
            "available": True,
        }
        # Probes are deliberately separate from discovery: /models can say a
        # model exists but not which request parameters it accepts. Retain the
        # last successful live probe until the dedicated probe refreshes it.
        previous = previous_catalog.get(key, {}) if isinstance(previous_catalog, dict) else {}
        if isinstance(previous, dict) and "reasoning_effort" in previous:
            catalog[key]["reasoning_effort"] = previous["reasoning_effort"]
    for entry in AZURE_MODELS:
        catalog[f"azure:{entry['id']}"] = {
            "tier": entry["tier"],
            "rank": entry["rank"],
            "capabilities": entry["capabilities"],
            "family": "azure-fallback",
            "available": "configured",
        }
    config["model_catalog"] = catalog
    # Retain the old flat field for callers that inspect it.
    config["model_ranks"] = {key: value["rank"] for key, value in catalog.items()}
    return config


def _render(config: dict[str, Any]) -> str:
    return yaml.safe_dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False)


def models_missing_reasoning_probe(config: dict[str, Any], model_ids: list[str]) -> list[str]:
    """Return newly discovered models with no successful effort capability record."""
    catalog = config.get("model_catalog", {})
    if not isinstance(catalog, dict):
        return list(model_ids)
    missing: list[str] = []
    for model_id in model_ids:
        profile = catalog.get(f"kiconnect:{model_id}", {}).get("reasoning_effort", {})
        if not isinstance(profile, dict) or not profile.get("supported"):
            missing.append(model_id)
    return missing


def probe_new_reasoning_efforts(model_ids: list[str]) -> None:
    """Run the lightweight probe only for newly discovered model IDs."""
    if not model_ids:
        print("Reasoning effort probe: all discovered models already have a live record")
        return
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / "probe_reasoning_effort.py"), "--write", "--models", *model_ids]
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, timeout=540)
    if result.returncode:
        raise RuntimeError(f"reasoning effort probe failed for {model_ids}")


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(content)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _without_provider(entries: list[str], provider: str = "kiconnect") -> list[str]:
    prefix = f"{provider}:"
    return [entry.removeprefix(prefix) for entry in entries if entry.startswith(prefix)]


def github_variable_plan(config: dict[str, Any], model_ids: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Return (repository variables, environment variables) for Metaculus Actions."""
    routes = config["task_routing"]
    forecast = _without_provider(routes["forecast"])
    cheap = _without_provider(routes["agent_step"])
    if not forecast or not cheap:
        raise ValueError("Cannot sync GitHub variables without forecast and cheap KIconnect routes")
    repository = {"KICONNECT_CHAT_MODELS": ",".join(model_ids)}
    environment = {
        "KICONNECT_MODEL": cheap[0],
        "KICONNECT_MODEL_FALLBACKS": ",".join(cheap[1:]),
        "KICONNECT_CHEAP_MODEL": cheap[0],
        "KICONNECT_CHEAP_MODEL_FALLBACKS": ",".join(cheap[1:]),
        "KICONNECT_FORECAST_MODEL": forecast[0],
        "KICONNECT_FORECAST_MODEL_FALLBACKS": ",".join(forecast[1:]),
        "KICONNECT_HIGH_MODEL": forecast[0],
        "KICONNECT_HIGH_MODEL_FALLBACKS": ",".join(forecast[1:]),
        "BOT_ENABLE_REASONING": "true",
        "BOT_CHEAP_REASONING_EFFORT": "low",
        "BOT_FORECAST_REASONING_EFFORT": "high",
    }
    return repository, environment


def _gh_json(args: list[str]) -> list[dict[str, str]]:
    # The server's general .env contains legacy GitHub tokens. Prefer the
    # authenticated gh keyring/config, which is known to have Actions scope.
    gh_env = {key: value for key, value in os.environ.items() if key not in {"GH_TOKEN", "GITHUB_TOKEN"}}
    result = subprocess.run(
        ["gh", *args], check=True, capture_output=True, text=True, timeout=30, env=gh_env
    )
    value = json.loads(result.stdout or "[]")
    return value if isinstance(value, list) else []


def sync_github_variables(config: dict[str, Any], model_ids: list[str]) -> None:
    """Update model-only Actions variables, skipping values that did not change."""
    if shutil.which("gh") is None:
        raise RuntimeError("gh CLI is required for --sync-github")
    repository_plan, environment_plan = github_variable_plan(config, model_ids)
    gh_env = {key: value for key, value in os.environ.items() if key not in {"GH_TOKEN", "GITHUB_TOKEN"}}
    current_repository = {
        item["name"]: item["value"]
        for item in _gh_json(["variable", "list", "--repo", GITHUB_REPO, "--json", "name,value"])
    }
    current_environment = {
        item["name"]: item["value"]
        for item in _gh_json(
            [
                "variable",
                "list",
                "--repo",
                GITHUB_REPO,
                "--env",
                GITHUB_ENVIRONMENT,
                "--json",
                "name,value",
            ]
        )
    }

    changed: list[str] = []
    for name, value in repository_plan.items():
        if current_repository.get(name) == value:
            continue
        subprocess.run(
            ["gh", "variable", "set", name, "--repo", GITHUB_REPO, "--body", value],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env=gh_env,
        )
        changed.append(name)
    for name, value in environment_plan.items():
        if current_environment.get(name) == value:
            continue
        subprocess.run(
            [
                "gh",
                "variable",
                "set",
                name,
                "--repo",
                GITHUB_REPO,
                "--env",
                GITHUB_ENVIRONMENT,
                "--body",
                value,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env=gh_env,
        )
        changed.append(name)
    print(f"GitHub Actions variables synced ({len(changed)} changed): {changed}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="print the generated config without writing")
    parser.add_argument(
        "--sync-github",
        action="store_true",
        help="also synchronize Metaculus GitHub Actions model variables",
    )
    parser.add_argument(
        "--probe-new",
        action="store_true",
        help="live-probe reasoning effort only for newly discovered KI Connect models",
    )
    args = parser.parse_args(argv)

    _load_env()
    api_key = os.environ.get("KICONNECT_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    if not api_key:
        print("ERROR: KICONNECT_API_KEY not set", file=sys.stderr)
        return 1

    try:
        model_ids = fetch_kiconnect_text_models(api_key)
        current = yaml.safe_load(ROUTING_YAML.read_text(encoding="utf-8")) or {}
        now = datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()
        config = build_config(current, model_ids, now)
        rendered = _render(config)
        # Parse the exact bytes before replacing the last-known-good config.
        yaml.safe_load(rendered)
    except Exception as exc:
        print(f"ERROR refreshing routing: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(rendered, end="")
    else:
        _atomic_write(ROUTING_YAML, rendered)
        print(f"Updated {ROUTING_YAML}")
        print(f"KIconnect text models: {model_ids}")
        if args.probe_new:
            try:
                probe_new_reasoning_efforts(models_missing_reasoning_probe(config, model_ids))
            except Exception as exc:
                # Availability/routing refresh remains useful even if a new
                # model's capability probe is temporarily unavailable.
                print(f"WARNING probing reasoning effort: {exc}", file=sys.stderr)
        if args.sync_github:
            try:
                sync_github_variables(config, model_ids)
            except Exception as exc:
                print(f"ERROR syncing GitHub variables: {exc}", file=sys.stderr)
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
