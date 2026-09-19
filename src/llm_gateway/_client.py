"""OpenAI-compatible chat client with ordered, observable fallback."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Generator

import httpx

from llm_gateway._config import get_config, provider_credentials
from llm_gateway._router import resolve_candidates

logger = logging.getLogger("llm_gateway")

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}
_MODEL_UNAVAILABLE_STATUS = {404, 410}
_FALLBACK_ERROR_HINTS = (
    "quota",
    "rate limit",
    "rate_limit",
    "capacity",
    "overloaded",
    "temporarily unavailable",
    "model not found",
    "model_not_found",
    "does not exist",
    "deployment not found",
)

# Avoid hammering a model that just failed when a cron job makes many calls.
# This is intentionally process-local; the next independent job gets a fresh try.
_unavailable_until: dict[str, float] = {}


@dataclass
class _FallbackError(Exception):
    message: str
    provider_wide: bool = False
    cooldown_seconds: float = 30.0

    def __str__(self) -> str:
        return self.message


def _azure_cfg() -> dict:
    return get_config()["providers"].get("azure", {})


def _request_parts(
    provider: str,
    model: str,
    messages: list[dict],
    extra: dict[str, Any],
) -> tuple[str, str, dict[str, str], dict[str, Any]]:
    base_url, api_key = provider_credentials(provider)
    if not base_url or not api_key:
        raise _FallbackError(
            f"{provider}:{model} is not configured",
            provider_wide=True,
            cooldown_seconds=300,
        )

    if provider == "azure":
        config = _azure_cfg()
        api_version = config.get("api_version", "2025-01-01-preview")
        if config.get("deployments_in_path"):
            url = f"{base_url}/openai/deployments/{model}/chat/completions"
            adjusted = dict(extra)
            if "max_tokens" in adjusted:
                adjusted["max_completion_tokens"] = adjusted.pop("max_tokens")
            body: dict[str, Any] = {"messages": messages, **adjusted}
        else:
            url = f"{base_url}/chat/completions"
            body = {"model": model, "messages": messages, **extra}
        params = {"api-version": api_version}
    else:
        url = f"{base_url}/chat/completions"
        params = {}
        adjusted = dict(extra)
        # KI Connect exposes GPT-5 deployments through the OpenAI API, which
        # rejects the legacy max_tokens field just like Azure does.  The local
        # free models still use max_tokens, so convert only this model family.
        if model.lower().startswith("gpt-5") and "gpt-oss" not in model.lower() and "max_tokens" in adjusted:
            adjusted["max_completion_tokens"] = adjusted.pop("max_tokens")
        body = {"model": model, "messages": messages, **adjusted}
    return url, api_key, params, body


def _fallback_error_from_response(provider: str, model: str, response: httpx.Response) -> _FallbackError | None:
    status = response.status_code
    text = response.text.lower()[:2000]
    retry_after = response.headers.get("retry-after", "").strip()
    try:
        cooldown = min(max(float(retry_after), 1.0), 3600.0) if retry_after else 30.0
    except ValueError:
        cooldown = 30.0

    if status == 401 or (status == 403 and not any(hint in text for hint in _FALLBACK_ERROR_HINTS)):
        # Authentication/authorization failures affect the whole provider.
        return _FallbackError(
            f"{provider}:{model} returned HTTP {status}",
            provider_wide=True,
            cooldown_seconds=max(cooldown, 300.0),
        )
    if status in {402, 403}:
        # Some gateways express a per-model quota as 402/403. Keep sibling
        # models eligible because the free pools often have separate quotas.
        return _FallbackError(
            f"{provider}:{model} returned quota HTTP {status}",
            cooldown_seconds=max(cooldown, 60.0),
        )
    if status in _RETRYABLE_STATUS or status in _MODEL_UNAVAILABLE_STATUS:
        return _FallbackError(
            f"{provider}:{model} returned HTTP {status}",
            cooldown_seconds=cooldown,
        )
    if status in {400, 422} and any(hint in text for hint in _FALLBACK_ERROR_HINTS):
        return _FallbackError(
            f"{provider}:{model} is unavailable ({status})",
            cooldown_seconds=max(cooldown, 60.0),
        )
    return None


def _validate_response(provider: str, model: str, payload: dict[str, Any]) -> None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise _FallbackError(f"{provider}:{model} returned no choices")
    message = choices[0].get("message") or {}
    if message.get("content") is None and not message.get("tool_calls"):
        raise _FallbackError(
            f"{provider}:{model} returned an empty completion",
            cooldown_seconds=5.0,
        )


def _call(
    provider: str,
    model: str,
    messages: list[dict],
    extra: dict[str, Any],
    timeout: float,
) -> dict:
    url, api_key, params, body = _request_parts(provider, model, messages, extra)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        response = httpx.post(url, params=params, headers=headers, json=body, timeout=timeout)
        # Some reasoning deployments reject non-default temperature.
        if response.status_code == 400 and "temperature" in response.text.lower() and "temperature" in body:
            body = {key: value for key, value in body.items() if key != "temperature"}
            response = httpx.post(url, params=params, headers=headers, json=body, timeout=timeout)
    except httpx.RequestError as exc:
        raise _FallbackError(f"{provider}:{model} request failed: {type(exc).__name__}") from exc

    fallback_error = _fallback_error_from_response(provider, model, response)
    if fallback_error is not None:
        raise fallback_error
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError as exc:
        raise _FallbackError(f"{provider}:{model} returned invalid JSON") from exc
    _validate_response(provider, model, payload)
    return payload


def _mark_unavailable(provider: str, model: str, seconds: float) -> None:
    _unavailable_until[f"{provider}:{model}"] = time.monotonic() + seconds


def _is_temporarily_unavailable(provider: str, model: str) -> bool:
    key = f"{provider}:{model}"
    until = _unavailable_until.get(key, 0.0)
    if until <= time.monotonic():
        _unavailable_until.pop(key, None)
        return False
    return True


def _candidate_extra(
    extra: dict[str, Any], task: str | None, provider: str, model: str
) -> dict[str, Any]:
    """Choose an effort valid for this specific fallback candidate.

    A task expresses an intent (for example ``low`` for job matching), while
    KI Connect models expose incompatible effort vocabularies.  The routing
    catalog records the live-probed translation for each model.  This prevents
    a perfectly healthy fallback from failing with a 400 merely because the
    preceding model used a different effort scale.
    """
    adjusted = dict(extra)
    cfg = get_config()
    policy = cfg.get("reasoning_effort_policy", {})
    requested = adjusted.get("reasoning_effort")
    target = requested or policy.get("task_targets", {}).get(task or "default")
    if not target:
        return adjusted

    entry = cfg.get("model_catalog", {}).get(f"{provider}:{model}", {})
    profile = entry.get("reasoning_effort", {}) if isinstance(entry, dict) else {}
    mapping = profile.get("task_target_map", {}) if isinstance(profile, dict) else {}
    selected = mapping.get(target)
    if selected:
        adjusted["reasoning_effort"] = selected
    elif requested is None:
        # No live capability record yet: do not invent a provider-specific
        # parameter. The caller still gets a normal completion and fallback.
        adjusted.pop("reasoning_effort", None)
    return adjusted


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
    """Send a chat request and immediately fail over on transient/model errors."""
    extra = {"max_tokens": max_tokens, "temperature": temperature, **kwargs}
    candidates = resolve_candidates(task=task, model=model)
    failures: list[str] = []
    failed_providers: set[str] = set()

    for provider, model_id in candidates:
        if provider in failed_providers or _is_temporarily_unavailable(provider, model_id):
            continue
        try:
            payload = _call(
                provider,
                model_id,
                messages,
                _candidate_extra(extra, task, provider, model_id),
                timeout,
            )
            if failures:
                logger.info("LLM fallback succeeded with %s:%s for task=%s", provider, model_id, task)
            return payload
        except _FallbackError as exc:
            failures.append(str(exc))
            _mark_unavailable(provider, model_id, exc.cooldown_seconds)
            if exc.provider_wide:
                failed_providers.add(provider)
            logger.warning("LLM fallback: %s", exc)
            continue

    detail = "; ".join(failures[-4:]) or "no configured candidates"
    raise RuntimeError(f"All LLM candidates exhausted for task={task!r}: {detail}")


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
    """Stream text, falling back only before the first chunk is emitted.

    Once output has been yielded, restarting on another model would duplicate or
    splice text, so a mid-stream failure is surfaced to the caller.
    """
    extra = {"max_tokens": max_tokens, "temperature": temperature, "stream": True, **kwargs}
    candidates = resolve_candidates(task=task, model=model)
    failures: list[str] = []
    failed_providers: set[str] = set()

    for provider, model_id in candidates:
        if provider in failed_providers or _is_temporarily_unavailable(provider, model_id):
            continue
        yielded = False
        try:
            url, api_key, params, body = _request_parts(
                provider, model_id, messages, _candidate_extra(extra, task, provider, model_id)
            )
            with httpx.stream(
                "POST",
                url,
                params=params,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=body,
                timeout=timeout,
            ) as response:
                fallback_error = _fallback_error_from_response(provider, model_id, response)
                if fallback_error is not None:
                    raise fallback_error
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yielded = True
                        yield content
                if not yielded:
                    raise _FallbackError(
                        f"{provider}:{model_id} returned an empty stream",
                        cooldown_seconds=5.0,
                    )
                return
        except _FallbackError as exc:
            if yielded:
                raise RuntimeError(f"Streaming failed after output began on {provider}:{model_id}") from exc
            failures.append(str(exc))
            _mark_unavailable(provider, model_id, exc.cooldown_seconds)
            if exc.provider_wide:
                failed_providers.add(provider)
            continue
        except httpx.RequestError as exc:
            if yielded:
                raise RuntimeError(f"Streaming failed after output began on {provider}:{model_id}") from exc
            failures.append(f"{provider}:{model_id} request failed: {type(exc).__name__}")
            _mark_unavailable(provider, model_id, 30.0)
            continue

    detail = "; ".join(failures[-4:]) or "no configured candidates"
    raise RuntimeError(f"All streaming LLM candidates exhausted for task={task!r}: {detail}")
