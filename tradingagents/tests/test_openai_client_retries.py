import unittest

from tradingagents.llm_clients.openai_client import (
    _is_retriable_openai_compatible_payload,
    _is_retriable_provider_value_error,
)


class TestOpenAIClientRetryHelpers(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
