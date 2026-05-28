"""Core chat() — route → try → 429/503 fallback → return OpenAI-format response."""

from __future__ import annotations

from typing import Any, Generator

import httpx

from llm_gateway._config import get_config, provider_credentials
from llm_gateway._router import resolve_candidates

_RETRYABLE = {429, 503, 529}


def _azure_api_version() -> str:
    cfg = get_config()
    return cfg["providers"].get("azure", {}).get("api_version", "2025-01-01-preview")


def _call(
    provider: str,
    model: str,
    messages: list[dict],
    extra: dict[str, Any],
    timeout: float,
) -> dict:
    base_url, api_key = provider_credentials(provider)
    if not api_key:
        raise ValueError(f"No API key for provider '{provider}'")

    params = {"api-version": _azure_api_version()} if provider == "azure" else {}

    resp = httpx.post(
        f"{base_url}/chat/completions",
        params=params,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, **extra},
        timeout=timeout,
    )
    if resp.status_code in _RETRYABLE:
        raise _QuotaError(f"{provider}:{model} → HTTP {resp.status_code}")
    resp.raise_for_status()
    return resp.json()


class _QuotaError(Exception):
    pass


def chat(
    messages: list[dict],
    *,
    task: str | None = None,
    model: str | None = None,
    max_tokens: int = 800,
    temperature: float = 0.7,
    timeout: float = 60.0,
    **kwargs: Any,
) -> dict:
    """Send a chat request, auto-routing by task with 429/503 fallback.

    Extra kwargs (e.g. response_format, top_p) are forwarded to the API.

    Args:
        messages:    OpenAI-format message list.
        task:        Routing hint ("cold_email", "code", "blog", "summarize", …).
        model:       Optional preference ("gpt-oss-120b" or "kiconnect:gpt-5.2").
        max_tokens, temperature, timeout: passed to the API.
        **kwargs:    Any extra OpenAI-compatible params (response_format, etc.).

    Returns:
        OpenAI-format response dict.

    Raises:
        RuntimeError: all candidates exhausted.
    """
    extra = {"max_tokens": max_tokens, "temperature": temperature, **kwargs}
    candidates = resolve_candidates(task=task, model=model)
    last_err: Exception | None = None

    for provider, mid in candidates:
        try:
            return _call(provider, mid, messages, extra, timeout)
        except _QuotaError as e:
            last_err = e
            continue
        except Exception:
            raise

    raise RuntimeError(
        f"All LLM candidates exhausted for task={task!r}. Last error: {last_err}"
    )


def stream_chat(
    messages: list[dict],
    *,
    task: str | None = None,
    model: str | None = None,
    max_tokens: int = 800,
    temperature: float = 0.7,
    timeout: float = 60.0,
    **kwargs: Any,
) -> Generator[str, None, None]:
    """Streaming version: yields text chunks. Falls back on quota errors."""
    import json

    extra = {"max_tokens": max_tokens, "temperature": temperature, "stream": True, **kwargs}
    candidates = resolve_candidates(task=task, model=model)

    for provider, mid in candidates:
        base_url, api_key = provider_credentials(provider)
        if not api_key:
            continue
        params = {"api-version": _azure_api_version()} if provider == "azure" else {}
        try:
            with httpx.stream(
                "POST",
                f"{base_url}/chat/completions",
                params=params,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": mid, "messages": messages, **extra},
                timeout=timeout,
            ) as resp:
                if resp.status_code in _RETRYABLE:
                    continue
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunk = json.loads(line[6:])
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
                return
        except Exception:
            continue

    raise RuntimeError(f"All LLM candidates exhausted for task={task!r}")
