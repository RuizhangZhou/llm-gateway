# LLM Gateway

Shared task-aware routing for the jobs on this server. Applications choose a
task, not a hard-coded model:

```python
from llm_gateway import chat

response = chat(messages, task="job_score")
```

`config/routing.yaml` is generated from KIconnect's live `/models` endpoint.
The refresh job runs every six hours and keeps the previous file if discovery
fails or returns no text models. It commits and pushes the config only when
model or routing data changes; timestamp-only refreshes leave the file alone.
The systemd job also synchronizes the model-only GitHub Actions variables used
by `RuizhangZhou/metaculus-bot`.

By default, gateway clients check
`RuizhangZhou/llm-gateway`'s `master/config/routing.yaml` on startup and every
five minutes. The last successfully fetched file is cached under
`~/.cache/llm-gateway/routing.yaml`; if GitHub is unreachable, clients continue
with that cache or the checked-out/package config on a fresh machine. Set
`LLM_GATEWAY_CONFIG_REPO`, `LLM_GATEWAY_CONFIG_REF`,
`LLM_GATEWAY_CONFIG_PATH`, `LLM_GATEWAY_CONFIG_REFRESH_SECONDS`, or
`LLM_GATEWAY_CONFIG_URL` to change the source or polling interval. For a private
repository, provide a read-only GitHub token as `LLM_GATEWAY_GITHUB_TOKEN`
(contents: read). Each runtime keeps provider credentials such as
`KICONNECT_API_KEY` and `AZURE_OPENAI_API_KEY` in its own environment; these
are never part of the shared routing file. The refresh host needs GitHub CLI
authentication with contents: write access to this repository so it can push
changed routing data. `LLM_GATEWAY_CONFIG_GIT_REMOTE` and
`LLM_GATEWAY_CONFIG_GIT_BRANCH` select the push target (defaults: `origin` and
`master`).

## Routing policy

| Work | Policy | Current order |
| --- | --- | --- |
| Posts / essays / simple summaries | free-first, fast writing | Mistral → Qwen → GPT-OSS |
| YouTube digest / job scoring / CV / email | free-first, multilingual and structured | Qwen → Mistral → GPT-OSS |
| Forecast / technical / code / complex agent planning | quality-first | newest GPT-5.x → GPT-OSS → Mistral → Qwen → remaining limited models |

Every chain ends with the remaining live KIconnect models and then the Azure
cross-provider fallback. New GPT-5.x versions are sorted by version
automatically. Unknown model families are retained at the end instead of being
silently promoted.

Fallback occurs on rate limits, quota/capacity errors, timeouts, connection
errors, HTTP 5xx, and removed models. Authentication errors skip the rest of
that provider. A short process-local cooldown prevents batch jobs from hitting
the same failed model for every item. Streaming only fails over before its first
output chunk, because switching afterward would splice two answers together.

## Refresh and inspection

```bash
uv run python scripts/refresh_routing.py --dry-run
uv run python scripts/refresh_routing.py
uv run python scripts/refresh_routing.py --sync-github
uv run python scripts/refresh_routing.py --publish-github
systemctl status llm-gateway-refresh.timer
```

The generated `model_catalog` records availability, tier, family, capabilities,
and rank. Family rules and task policies live in `scripts/refresh_routing.py`.

## Agent design

Keep search orchestration separate from the model router:

1. Retrieve with deterministic APIs/search and cache raw results.
2. Extract claims, dates, URLs, and other evidence into a schema.
3. Use `agent_step` for cheap per-document classification and scoring.
4. Use `agent_complex` only for cross-source synthesis or technical judgment.
5. Validate the structured result, then require human approval for external
   actions such as applications, posts, or forecast submissions.

This makes the search process testable and lets cheap models do most work while
reserving limited models for the step where they materially improve quality.
