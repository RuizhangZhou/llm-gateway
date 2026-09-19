"""Server-wide, task-aware LLM routing with automatic fallback.

Usage::

    from llm_gateway import chat

    # Cheap tasks use free models first; forecast/technical tasks use frontier first.
    resp = chat(messages=[{"role": "user", "content": "..."}], task="cold_email")
    print(resp["choices"][0]["message"]["content"])

    # With extra OpenAI params (forwarded as-is)
    resp = chat(messages=[...], task="blog", response_format={"type": "json_object"})
"""

from llm_gateway._client import chat, stream_chat

__all__ = ["chat", "stream_chat"]
