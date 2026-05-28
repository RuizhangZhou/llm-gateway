"""Task → ordered (provider, model) list with availability filtering."""

from __future__ import annotations

from llm_gateway._config import get_config, kiconnect_available_models


def resolve_candidates(
    task: str | None = None,
    model: str | None = None,
) -> list[tuple[str, str]]:
    """Return ordered list of (provider, model_id) to try for this request.

    If `model` is given (bare name or "provider:model"), that model is tried
    first, followed by the task fallback chain.

    Returns at least one candidate.
    """
    cfg = get_config()
    task_routing: dict[str, list[str]] = cfg.get("task_routing", {})
    available_kiconnect = set(kiconnect_available_models())

    # Parse "provider:model" or bare "model" → list of strings
    raw: list[str] = []
    if model:
        raw.append(model)
    raw.extend(task_routing.get(task or "default", task_routing["default"]))
    # Always append the full default chain as ultimate fallback
    for entry in task_routing.get("default", []):
        if entry not in raw:
            raw.append(entry)

    # Deduplicate while preserving order
    seen: set[str] = set()
    ordered: list[tuple[str, str]] = []
    for entry in raw:
        if ":" in entry:
            provider, mid = entry.split(":", 1)
        else:
            provider, mid = "kiconnect", entry

        key = f"{provider}:{mid}"
        if key in seen:
            continue
        seen.add(key)

        # Skip KIconnect models not currently available
        if provider == "kiconnect" and mid not in available_kiconnect:
            continue

        ordered.append((provider, mid))

    # If everything was filtered out, return the first entry unconditionally
    if not ordered and raw:
        first = raw[0]
        if ":" in first:
            provider, mid = first.split(":", 1)
        else:
            provider, mid = "kiconnect", first
        ordered.append((provider, mid))

    return ordered
