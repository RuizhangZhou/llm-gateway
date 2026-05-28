#!/usr/bin/env python3
"""Weekly routing table refresh.

Reads /root/.openclaw/kiconnect/quota.json (maintained by kiconnect-refresh.timer)
and rebuilds config/routing.yaml with up-to-date model availability + rankings.

Run via cron or systemd timer after kiconnect-refresh completes.
Usage: python scripts/refresh_routing.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

QUOTA_JSON = Path("/root/.openclaw/kiconnect/quota.json")
ROUTING_YAML = Path(__file__).parent.parent / "config" / "routing.yaml"

# Capability ranking: higher = stronger / preferred for quality-sensitive tasks
_KICONNECT_RANKS = {
    "gpt-oss-120b": 100,
    "gpt-5.2": 90,
    "gpt-5.4-mini": 70,
    "mistral-small-4-119b-2603": 60,
}

_AZURE_MODELS = [
    {"id": "gpt-4o", "rank": 85},
]


def load_kiconnect_text_models() -> list[dict]:
    """Return available KIconnect text-generation models sorted by rank desc."""
    if not QUOTA_JSON.exists():
        print(f"WARNING: {QUOTA_JSON} not found — using hardcoded fallback", file=sys.stderr)
        return [{"id": mid, "rank": rank} for mid, rank in sorted(_KICONNECT_RANKS.items(), key=lambda x: -x[1])]

    data = json.loads(QUOTA_JSON.read_text(encoding="utf-8"))
    models_meta = data.get("models", {})
    result = []
    for mid, meta in models_meta.items():
        if not isinstance(meta, dict):
            continue
        if meta.get("category", "chat") != "chat":
            continue
        if meta.get("availableNow", True) is False:
            continue
        if meta.get("useByDefault", True) is False:
            continue
        rank = _KICONNECT_RANKS.get(mid, 50)
        result.append({"id": mid, "rank": rank})
    result.sort(key=lambda x: -x["rank"])
    return result


def build_task_routing(kc_models: list[dict]) -> dict:
    """Rebuild task_routing based on current available models."""
    kc_ids = [f"kiconnect:{m['id']}" for m in kc_models]
    az_ids = [f"azure:{m['id']}" for m in _AZURE_MODELS]

    # Best KIconnect model for quality tasks (rank >= 85)
    quality_kc = [f"kiconnect:{m['id']}" for m in kc_models if m["rank"] >= 85]
    # Fast/cheap KIconnect models (rank < 85)
    fast_kc = [f"kiconnect:{m['id']}" for m in kc_models if m["rank"] < 85]

    def chain(*lists):
        seen = set()
        out = []
        for lst in lists:
            for item in lst:
                if item not in seen:
                    seen.add(item)
                    out.append(item)
        return out

    return {
        "cold_email":  chain(quality_kc[:1], kc_ids[1:2], az_ids),
        "code":        chain(kc_ids[:1], az_ids, kc_ids[1:]),
        "summarize":   chain(fast_kc, quality_kc),
        "classify":    chain(fast_kc[:1], fast_kc[1:2]),
        "default":     chain(kc_ids, az_ids),
    }


def build_model_ranks(kc_models: list[dict]) -> dict:
    ranks = {}
    for m in kc_models:
        ranks[f"kiconnect:{m['id']}"] = m["rank"]
    for m in _AZURE_MODELS:
        ranks[f"azure:{m['id']}"] = m["rank"]
    return ranks


def main() -> int:
    kc_models = load_kiconnect_text_models()
    if not kc_models:
        print("ERROR: no KIconnect text models found", file=sys.stderr)
        return 1

    # Load current YAML to preserve provider config + comments structure
    current = yaml.safe_load(ROUTING_YAML.read_text(encoding="utf-8"))
    current["task_routing"] = build_task_routing(kc_models)
    current["model_ranks"] = build_model_ranks(kc_models)

    ROUTING_YAML.write_text(
        f"# Auto-updated by scripts/refresh_routing.py at "
        f"{datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()}\n"
        + yaml.dump(current, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    print(f"Updated {ROUTING_YAML}")
    print(f"KIconnect text models: {[m['id'] for m in kc_models]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
