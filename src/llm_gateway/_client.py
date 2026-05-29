"""Core chat() — route → try → 429/503 fallback → return OpenAI-format response."""

from __future__ import annotations

from typing import Any, Generator

import httpx

from llm_gateway._config import get_config, provider_credentials
from llm_gateway._router import resolve_candidates

_RETRYABLE = {429, 503, 529}


def _azure_cfg() -> dict:
    return get_config()["providers"].get("azure", {})


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

    if provider == "azure":
        acfg = _azure_cfg()
        api_version = acfg.get("api_version", "2025-01-01-preview")
        if acfg.get("deployments_in_path"):
            # openai.azure.com: /openai/deployments/{deployment}/chat/completions
            # Newer models (gpt-5.x) use max_completion_tokens instead of max_tokens
            url = f"{base_url}/openai/deployments/{model}/chat/completions"
            adjusted = dict(extra)
            if "max_tokens" in adjusted:
                adjusted["max_completion_tokens"] = adjusted.pop("max_tokens")
            body: dict[str, Any] = {"messages": messages, **adjusted}
        else:
            # cognitiveservices.azure.com/openai/v1 style: flat /chat/completions
            url = f"{base_url}/chat/completions"
            body = {"model": model, "messages": messages, **extra}
        params = {"api-version": api_version}
    else:
        url = f"{base_url}/chat/completions"
        params = {}
        body = {"model": model, "messages": messages, **extra}

    resp = httpx.post(
        url,
        params=params,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body,
        timeout=timeout,
    )
    # Some reasoning models (gpt-5.3-chat) reject temperature != default — retry without it
    if resp.status_code == 400 and b'"temperature"' in resp.content and "temperature" in body:
        body_no_temp = {k: v for k, v in body.items() if k != "temperature"}
        resp = httpx.post(
            url,
            params=params,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=body_no_temp,
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
        if provider == "azure":
            acfg = _azure_cfg()
            api_version = acfg.get("api_version", "2025-01-01-preview")
            if acfg.get("deployments_in_path"):
                stream_url = f"{base_url}/openai/deployments/{mid}/chat/completions"
                adjusted_extra = dict(extra)
                if "max_tokens" in adjusted_extra:
                    adjusted_extra["max_completion_tokens"] = adjusted_extra.pop("max_tokens")
                stream_body: dict[str, Any] = {"messages": messages, **adjusted_extra}
            else:
                stream_url = f"{base_url}/chat/completions"
                stream_body = {"model": mid, "messages": messages, **extra}
            params = {"api-version": api_version}
        else:
            stream_url = f"{base_url}/chat/completions"
            params = {}
            stream_body = {"model": mid, "messages": messages, **extra}
        try:
            with httpx.stream(
                "POST",
                stream_url,
                params=params,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=stream_body,
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
