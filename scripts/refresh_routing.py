#!/usr/bin/env python3
"""Weekly: fetch live KIconnect model list and rebuild config/routing.yaml.

Run after kiconnect-refresh.timer (Mon 06:30 UTC).
Usage: uv run python scripts/refresh_routing.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROUTING_YAML = Path("/root/llm-gateway/config/routing.yaml")
KICONNECT_BASE_URL = "https://chat.kiconnect.nrw/api/v1"
ENV_FILE = Path("/root/.env")

# Capability rank: higher = stronger / used first for quality tasks.
# Update this list when KIconnect adds or retires models.
MODEL_RANKS: dict[str, int] = {
    "gpt-oss-120b": 100,
    "gpt-5.2": 90,
    "gpt-5.4-mini": 70,
    "mistral-small-4-119b-2603": 60,
}

AZURE_MODELS = [("gpt-4o", 85)]

EMBEDDING_HINTS = {"embedding", "e5-"}


def _load_env() -> None:
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        if k and k not in os.environ:
            os.environ[k] = v.strip().strip('"').strip("'")


def fetch_kiconnect_text_models(api_key: str) -> list[tuple[str, int]]:
    """Return (model_id, rank) list for text-gen models, sorted by rank desc."""
    import urllib.request

    req = urllib.request.Request(
        f"{KICONNECT_BASE_URL}/models",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())

    results = []
    for item in data.get("data", []):
        mid = str(item.get("id", "")).strip()
        if not mid:
            continue
        if any(h in mid.lower() for h in EMBEDDING_HINTS):
            continue
        rank = MODEL_RANKS.get(mid, 50)
        results.append((mid, rank))

    results.sort(key=lambda x: -x[1])
    return results


def build_task_routing(kc: list[tuple[str, int]]) -> dict:
    kc_ids = [f"kiconnect:{mid}" for mid, _ in kc]
    az_ids = [f"azure:{mid}" for mid, _ in AZURE_MODELS]
    quality = [f"kiconnect:{mid}" for mid, rank in kc if rank >= 85]
    fast = [f"kiconnect:{mid}" for mid, rank in kc if rank < 85]

    def chain(*lists):
        seen, out = set(), []
        for lst in lists:
            for item in lst:
                if item not in seen:
                    seen.add(item); out.append(item)
        return out

    return {
        "cold_email": chain(quality[:1], kc_ids[1:2], az_ids),
        "blog":       chain(quality[:1], kc_ids[1:2], az_ids),
        "cv_tailor":  chain(quality[:1], kc_ids[1:2], az_ids),
        "code":       chain(kc_ids[:1], az_ids, kc_ids[1:]),
        "summarize":  chain(fast, quality),
        "classify":   chain(fast[:1], fast[1:2]),
        "default":    chain(kc_ids, az_ids),
    }


def main() -> int:
    _load_env()
    api_key = os.environ.get("KICONNECT_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    if not api_key:
        print("ERROR: KICONNECT_API_KEY not set", file=sys.stderr)
        return 1

    try:
        kc_models = fetch_kiconnect_text_models(api_key)
    except Exception as e:
        print(f"ERROR fetching KIconnect models: {e}", file=sys.stderr)
        return 1

    current = yaml.safe_load(ROUTING_YAML.read_text(encoding="utf-8"))
    current["task_routing"] = build_task_routing(kc_models)
    current["model_ranks"] = {
        **{f"kiconnect:{mid}": rank for mid, rank in kc_models},
        **{f"azure:{mid}": rank for mid, rank in AZURE_MODELS},
    }

    now = datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()
    ROUTING_YAML.write_text(
        f"# Auto-updated {now}\n"
        + yaml.dump(current, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"Updated {ROUTING_YAML}")
    print(f"KIconnect models: {[mid for mid, _ in kc_models]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
