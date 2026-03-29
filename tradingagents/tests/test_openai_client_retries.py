import unittest

import httpx
import openai

from tradingagents.llm_clients.openai_client import (
    _is_retriable_openai_compatible_payload,
    _is_retriable_openai_sdk_error,
    _is_retriable_provider_value_error,
)


class TestOpenAIClientRetryHelpers(unittest.TestCase):
    def test_linear_backoff_defaults(self):
        import os

        os.environ.pop("LLM_RETRY_FIRST_WAIT_SEC", None)
        os.environ.pop("LLM_RETRY_STEP_SEC", None)
        import importlib

        import tradingagents.llm_clients.openai_client as oc

        importlib.reload(oc)
        self.assertAlmostEqual(oc._backoff_seconds(0), 60.0)
        self.assertAlmostEqual(oc._backoff_seconds(1), 90.0)
        self.assertAlmostEqual(oc._backoff_seconds(2), 120.0)

    def test_openrouter_502_payload_retriable(self):
        self.assertTrue(
            _is_retriable_openai_compatible_payload(
                {"message": "Provider returned error", "code": 502}
            )
        )

    def test_rate_limit_message_retriable(self):
        self.assertTrue(
            _is_retriable_openai_compatible_payload(
                {"message": "Rate limit exceeded", "code": 400}
            )
        )

    def test_benign_value_error_not_retriable(self):
        self.assertFalse(_is_retriable_provider_value_error(ValueError("wrong format")))

    def test_value_error_wrapping_payload(self):
        self.assertTrue(
            _is_retriable_provider_value_error(
                ValueError({"message": "Provider returned error", "code": 503})
            )
        )

    def test_sdk_rate_limit_error_retriable(self):
        req = httpx.Request("POST", "http://example.com/v1")
        resp = httpx.Response(429, request=req)
        err = openai.RateLimitError("rate limited", response=resp, body={})
        self.assertTrue(_is_retriable_openai_sdk_error(err))

    def test_sdk_connection_error_retriable(self):
        req = httpx.Request("POST", "http://example.com/v1")
        err = openai.APIConnectionError(message="reset", request=req)
        self.assertTrue(_is_retriable_openai_sdk_error(err))

    def test_sdk_auth_error_not_retriable(self):
        req = httpx.Request("POST", "http://example.com/v1")
        resp = httpx.Response(401, request=req)
        err = openai.AuthenticationError("nope", response=resp, body={})
        self.assertFalse(_is_retriable_openai_sdk_error(err))


if __name__ == "__main__":
    unittest.main()
