"""Tests for structured-output-only LLM binding (schemas.outputs path)."""

import os
import unittest
from unittest.mock import MagicMock, patch

from tradingagents.llm_clients.invoke_fallback import bound_llm_for_structured_output


class TestBoundLlmForStructuredOutput(unittest.TestCase):
    def test_binds_max_tokens_only_when_temp_unset(self):
        llm = MagicMock()
        llm.bind.return_value = llm
        real_getenv = os.getenv

        def getenv_no_struct_temp(key: str, default=None):
            if key == "LLM_STRUCTURED_TEMPERATURE":
                return None
            return real_getenv(key, default)

        with patch.dict(os.environ, {"LLM_STRUCTURED_MAX_TOKENS": "2048"}):
            with patch(
                "tradingagents.llm_clients.invoke_fallback.os.getenv",
                side_effect=getenv_no_struct_temp,
            ):
                bound_llm_for_structured_output(llm)
        llm.bind.assert_called_once_with(max_tokens=2048)

    def test_binds_temperature_with_max_tokens(self):
        llm = MagicMock()
        llm.configure_mock(model_name=None, model="qwen/test")
        llm.bind.return_value = llm
        with patch.dict(
            os.environ,
            {
                "LLM_STRUCTURED_MAX_TOKENS": "4096",
                "LLM_STRUCTURED_TEMPERATURE": "0.15",
            },
        ):
            bound_llm_for_structured_output(llm)
        llm.bind.assert_called_once_with(max_tokens=4096, temperature=0.15)

    def test_skips_temperature_for_gpt5_model(self):
        llm = MagicMock()
        llm.configure_mock(model_name=None, model="gpt-5-mini")
        llm.bind.return_value = llm
        with patch.dict(
            os.environ,
            {
                "LLM_STRUCTURED_MAX_TOKENS": "8192",
                "LLM_STRUCTURED_TEMPERATURE": "0",
            },
        ):
            bound_llm_for_structured_output(llm)
        llm.bind.assert_called_once_with(max_tokens=8192)


if __name__ == "__main__":
    unittest.main()
