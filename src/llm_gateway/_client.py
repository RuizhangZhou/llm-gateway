"""Core chat() function: route → try → fallback → return OpenAI-format response."""

from __future__ import annotations

import httpx

from llm_gateway._config import get_config, provider_credentials
from llm_gateway._router import resolve_candidates
from llm_gateway._usage import record

# HTTP status codes that indicate temporary quota/rate-limit → retry with next model
_RETRYABLE = {429, 503, 529}


def _azure_api_version(base_url: str) -> str:
    cfg = get_config()
    return cfg["providers"].get("azure", {}).get("api_version", "2025-01-01-preview")


def _call_provider(
    provider: str,
    model: str,
    messages: list[dict],
    *,
    max_tokens: int = 800,
    temperature: float = 0.7,
    timeout: float = 60.0,
) -> dict:
    """Make one OpenAI-compatible chat/completions request. Returns raw JSON dict."""
    base_url, api_key = provider_credentials(provider)
    if not api_key:
        raise ValueError(f"No API key for provider '{provider}'")

    params = {}
    if provider == "azure":
        params["api-version"] = _azure_api_version(base_url)

    resp = httpx.post(
        f"{base_url}/chat/completions",
        params=params,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
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
    app: str | None = None,
    max_tokens: int = 800,
    temperature: float = 0.7,
    timeout: float = 60.0,
) -> dict:
    """Send a chat request, routing to the best available model for the task.

    Falls back through the task's model list on quota / rate-limit errors (429, 503).

    Args:
        messages: OpenAI-format message list.
        task:     Task hint for routing (e.g. "cold_email", "code", "summarize").
        model:    Optional model preference ("gpt-oss-120b" or "kiconnect:gpt-5.2").
                  Tried first; task fallback chain applies if it fails.
        app:      Caller app name for usage tracking.
        max_tokens, temperature, timeout: passed to the API.

    Returns:
        OpenAI-format response dict.

    Raises:
        RuntimeError: if all candidates are exhausted.
    """
    candidates = resolve_candidates(task=task, model=model)
    last_error: Exception | None = None

    for provider, mid in candidates:
        try:
            resp = _call_provider(
                provider, mid, messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
            usage = resp.get("usage", {})
            record(
                provider=provider, model=mid,
                tokens_in=usage.get("prompt_tokens", 0),
                tokens_out=usage.get("completion_tokens", 0),
                task=task, app=app, success=True,
            )
            return resp
        except _QuotaError as e:
            record(provider=provider, model=mid, task=task, app=app, success=False)
            last_error = e
            continue
        except Exception as e:
            record(provider=provider, model=mid, task=task, app=app, success=False)
            raise

    raise RuntimeError(
        f"All LLM candidates exhausted for task={task!r}. Last error: {last_error}"
    )


def stream_chat(
    messages: list[dict],
    *,
    task: str | None = None,
    model: str | None = None,
    app: str | None = None,
    max_tokens: int = 800,
    temperature: float = 0.7,
    timeout: float = 60.0,
):
    """Streaming version: yields text chunks. Falls back on quota errors."""
    candidates = resolve_candidates(task=task, model=model)

    for provider, mid in candidates:
        base_url, api_key = provider_credentials(provider)
        if not api_key:
            continue
        params = {}
        if provider == "azure":
            params["api-version"] = _azure_api_version(base_url)
        try:
            with httpx.stream(
                "POST",
                f"{base_url}/chat/completions",
                params=params,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": mid,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": True,
                },
                timeout=timeout,
            ) as resp:
                if resp.status_code in _RETRYABLE:
                    record(provider=provider, model=mid, task=task, app=app, success=False)
                    continue
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        import json
                        chunk = json.loads(line[6:])
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
                return
        except Exception:
            record(provider=provider, model=mid, task=task, app=app, success=False)
            continue

    raise RuntimeError(f"All LLM candidates exhausted for task={task!r}")
