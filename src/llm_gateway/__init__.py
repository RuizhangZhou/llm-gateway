"""llm-gateway: server-wide LLM routing with task-based model selection and 429 fallback.

Usage::

    from llm_gateway import chat

    # Task-based routing: cold_email → gpt-oss-120b → gpt-5.2 → azure:gpt-4o
    resp = chat(messages=[{"role": "user", "content": "..."}], task="cold_email")
    print(resp["choices"][0]["message"]["content"])

    # With extra OpenAI params (forwarded as-is)
    resp = chat(messages=[...], task="blog", response_format={"type": "json_object"})
"""

from llm_gateway._client import chat, stream_chat

__all__ = ["chat", "stream_chat"]
