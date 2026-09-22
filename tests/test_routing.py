from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from llm_gateway import _client


def _load_refresh_module():
    path = Path(__file__).parents[1] / "scripts" / "refresh_routing.py"
    spec = importlib.util.spec_from_file_location("refresh_routing", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


refresh = _load_refresh_module()


class RoutingPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.models = [
            "mistral-small-4-119b-2603",
            "qwen3.8-27b",
            "gpt-oss-120b",
            "gpt-5.4-mini",
            "gpt-5.5",
        ]

    def test_free_first_routes_are_task_specific(self) -> None:
        routes = refresh.build_task_routing(self.models)
        self.assertEqual(routes["blog"][:3], [
            "kiconnect:mistral-small-4-119b-2603",
            "kiconnect:qwen3.8-27b",
            "kiconnect:gpt-oss-120b",
        ])
        self.assertEqual(routes["job_score"][:3], [
            "kiconnect:qwen3.8-27b",
            "kiconnect:mistral-small-4-119b-2603",
            "kiconnect:gpt-oss-120b",
        ])
        self.assertEqual(routes["job_score"][3:5], [
            "kiconnect:gpt-5.4-mini",
            "kiconnect:gpt-5.5",
        ])

    def test_quality_route_automatically_promotes_new_gpt_version(self) -> None:
        routes = refresh.build_task_routing([*self.models, "gpt-5.6"])
        self.assertEqual(routes["forecast"][:4], [
            "kiconnect:gpt-5.6",
            "kiconnect:gpt-oss-120b",
            "kiconnect:mistral-small-4-119b-2603",
            "kiconnect:qwen3.8-27b",
        ])

    def test_unknown_family_is_never_silently_promoted(self) -> None:
        routes = refresh.build_task_routing([*self.models, "new-mystery-model"])
        self.assertGreater(routes["default"].index("kiconnect:new-mystery-model"), 2)

    def test_instant_route_is_available_for_latency_critical_calls(self) -> None:
        routes = refresh.build_task_routing(self.models)
        self.assertEqual(routes["instant"][0], "kiconnect:mistral-small-4-119b-2603")

    def test_existing_probe_is_preserved_during_model_refresh(self) -> None:
        current = {
            "model_catalog": {
                "kiconnect:gpt-5.5": {"reasoning_effort": {"supported": ["none", "high"]}}
            }
        }
        config = refresh.build_config(current, self.models, "2026-09-19T00:00:00+00:00")
        self.assertEqual(
            config["model_catalog"]["kiconnect:gpt-5.5"]["reasoning_effort"]["supported"], ["none", "high"]
        )
        self.assertEqual(refresh.models_missing_reasoning_probe(config, self.models), [
            "mistral-small-4-119b-2603", "qwen3.8-27b", "gpt-oss-120b", "gpt-5.4-mini"
        ])

    def test_github_plan_uses_same_routes(self) -> None:
        config = {"task_routing": refresh.build_task_routing(self.models)}
        repository, environment = refresh.github_variable_plan(config, self.models)
        self.assertIn("qwen3.8-27b", repository["KICONNECT_CHAT_MODELS"])
        self.assertEqual(environment["KICONNECT_CHEAP_MODEL"], "qwen3.8-27b")
        self.assertEqual(environment["KICONNECT_FORECAST_MODEL"], "gpt-5.5")
        self.assertTrue(environment["KICONNECT_FORECAST_MODEL_FALLBACKS"].startswith("gpt-oss-120b"))

    def test_github_plan_skips_excluded_models(self) -> None:
        config = {"task_routing": refresh.build_task_routing(self.models)}
        with patch.object(refresh, "GITHUB_EXCLUDED_MODELS", {"qwen3.8-27b"}):
            repository, environment = refresh.github_variable_plan(config, self.models)
        self.assertIn("qwen3.8-27b", repository["KICONNECT_CHAT_MODELS"])
        self.assertEqual(environment["KICONNECT_CHEAP_MODEL"], "mistral-small-4-119b-2603")
        self.assertNotIn("qwen3.8-27b", ",".join(environment.values()))

    def test_github_plan_skips_models_rejecting_the_chain_effort(self) -> None:
        def efforts(*supported: str) -> dict:
            return {"reasoning_effort": {"supported": list(supported)}}

        config = {
            "task_routing": refresh.build_task_routing(self.models),
            "model_catalog": {
                "kiconnect:mistral-small-4-119b-2603": efforts("none", "high"),
                "kiconnect:qwen3.8-27b": efforts("none", "low", "medium", "xhigh"),
                "kiconnect:gpt-oss-120b": efforts("low", "medium", "high"),
                "kiconnect:gpt-5.4-mini": efforts("none", "low", "high"),
                "kiconnect:gpt-5.5": efforts("none", "low", "high", "xhigh"),
            },
        }
        _, environment = refresh.github_variable_plan(config, self.models)
        self.assertEqual(environment["KICONNECT_CHEAP_MODEL"], "qwen3.8-27b")
        self.assertTrue(environment["KICONNECT_CHEAP_MODEL_FALLBACKS"].startswith("gpt-oss-120b"))
        self.assertNotIn("mistral", environment["KICONNECT_CHEAP_MODEL_FALLBACKS"])
        # The forecast chain runs at "high", which qwen rejects.
        self.assertIn("mistral-small-4-119b-2603", environment["KICONNECT_FORECAST_MODEL_FALLBACKS"])
        self.assertNotIn("qwen3.8-27b", environment["KICONNECT_FORECAST_MODEL_FALLBACKS"])


class FallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        _client._unavailable_until.clear()

    def test_rate_limit_falls_through_to_next_model(self) -> None:
        candidates = [("kiconnect", "first"), ("kiconnect", "second")]
        successful = {"choices": [{"message": {"content": "ok"}}]}
        with patch.object(_client, "resolve_candidates", return_value=candidates), patch.object(
            _client,
            "_call",
            side_effect=[_client._FallbackError("rate limited"), successful],
        ) as call:
            result = _client.chat([{"role": "user", "content": "hi"}], task="test")
        self.assertEqual(result, successful)
        self.assertEqual(call.call_count, 2)

    def test_non_retryable_bad_request_is_not_classified_as_fallback(self) -> None:
        response = httpx.Response(400, text="invalid request schema")
        self.assertIsNone(_client._fallback_error_from_response("p", "m", response))

    def test_removed_model_is_classified_as_fallback(self) -> None:
        response = httpx.Response(404, text="model not found")
        error = _client._fallback_error_from_response("p", "m", response)
        self.assertIsInstance(error, _client._FallbackError)

    def test_quota_forbidden_does_not_disable_sibling_models(self) -> None:
        response = httpx.Response(403, text="model quota exhausted")
        error = _client._fallback_error_from_response("p", "m", response)
        self.assertIsInstance(error, _client._FallbackError)
        self.assertFalse(error.provider_wide)

    def test_empty_success_is_rejected(self) -> None:
        with self.assertRaises(_client._FallbackError):
            _client._validate_response("p", "m", {"choices": [{"message": {"content": None}}]})

    def test_effort_is_translated_for_the_fallback_model(self) -> None:
        config = {
            "reasoning_effort_policy": {"task_targets": {"job_score": "low"}},
            "model_catalog": {
                "kiconnect:mistral": {
                    "reasoning_effort": {"task_target_map": {"low": "none"}}
                },
                "kiconnect:oss": {
                    "reasoning_effort": {"task_target_map": {"low": "low"}}
                },
            },
        }
        with patch.object(_client, "get_config", return_value=config):
            self.assertEqual(
                _client._candidate_extra({}, "job_score", "kiconnect", "mistral")["reasoning_effort"],
                "none",
            )
            self.assertEqual(
                _client._candidate_extra({}, "job_score", "kiconnect", "oss")["reasoning_effort"], "low"
            )

    def test_high_intent_maps_to_each_free_models_maximum(self) -> None:
        config = {
            "reasoning_effort_policy": {"task_targets": {"job_score": "high"}},
            "model_catalog": {
                "kiconnect:mistral": {"reasoning_effort": {"task_target_map": {"high": "high"}}},
                "kiconnect:qwen": {"reasoning_effort": {"task_target_map": {"high": "xhigh"}}},
                "kiconnect:oss": {"reasoning_effort": {"task_target_map": {"high": "high"}}},
            },
        }
        with patch.object(_client, "get_config", return_value=config):
            self.assertEqual(_client._candidate_extra({}, "job_score", "kiconnect", "mistral")["reasoning_effort"], "high")
            self.assertEqual(_client._candidate_extra({}, "job_score", "kiconnect", "qwen")["reasoning_effort"], "xhigh")
            self.assertEqual(_client._candidate_extra({}, "job_score", "kiconnect", "oss")["reasoning_effort"], "high")

    def test_kiconnect_gpt5_uses_max_completion_tokens(self) -> None:
        with patch.object(_client, "provider_credentials", return_value=("https://example.test/v1", "key")):
            _, _, _, body = _client._request_parts(
                "kiconnect", "gpt-5.5", [{"role": "user", "content": "hi"}], {"max_tokens": 12}
            )
        self.assertNotIn("max_tokens", body)
        self.assertEqual(body["max_completion_tokens"], 12)


if __name__ == "__main__":
    unittest.main()
