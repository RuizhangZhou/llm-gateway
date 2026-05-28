"""Task → ordered (provider, model) list."""

from __future__ import annotations

from llm_gateway._config import get_config


def resolve_candidates(
    task: str | None = None,
    model: str | None = None,
) -> list[tuple[str, str]]:
    """Return ordered list of (provider, model_id) to try for this request.

    If `model` is given (bare name or "provider:model"), that model is tried
    first, followed by the task fallback chain.
    """
    cfg = get_config()
    task_routing: dict[str, list[str]] = cfg.get("task_routing", {})

    raw: list[str] = []
    if model:
        raw.append(model)
    raw.extend(task_routing.get(task or "default", task_routing.get("default", [])))
    for entry in task_routing.get("default", []):
        if entry not in raw:
            raw.append(entry)

    seen: set[str] = set()
    ordered: list[tuple[str, str]] = []
    for entry in raw:
        if ":" in entry:
            provider, mid = entry.split(":", 1)
        else:
            provider, mid = "kiconnect", entry
        key = f"{provider}:{mid}"
        if key not in seen:
            seen.add(key)
            ordered.append((provider, mid))

    return ordered
