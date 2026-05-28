"""llm-gateway: server-wide LLM routing with task-based model selection and quota fallback.

Usage::

    from llm_gateway import chat

    # Route automatically by task type
    response = chat(messages=[{"role": "user", "content": "..."}], task="cold_email")
    print(response["choices"][0]["message"]["content"])

    # Or specify a model hint (still falls back if unavailable)
    response = chat(messages=[...], model="gpt-oss-120b")
"""

from llm_gateway._client import chat, stream_chat

__all__ = ["chat", "stream_chat"]
