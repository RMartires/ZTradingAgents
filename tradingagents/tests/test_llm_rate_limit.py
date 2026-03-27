import unittest

from tradingagents.llm_clients.llm_rate_limit import (
    LLMRateLimiter,
    acquire_llm_slot,
    set_llm_rate_limit_rpm,
)


class TestLLMRateLimiter(unittest.TestCase):
    def tearDown(self) -> None:
        set_llm_rate_limit_rpm(None)

    def test_set_disables_global(self) -> None:
        set_llm_rate_limit_rpm(10)
        acquire_llm_slot()
        set_llm_rate_limit_rpm(None)
        for _ in range(5):
            acquire_llm_slot()

    def test_local_limiter_init(self) -> None:
        lim = LLMRateLimiter(5)
        self.assertEqual(lim.max_calls, 5)

    def test_global_integer_floor(self) -> None:
        set_llm_rate_limit_rpm(3.9)
        acquire_llm_slot()
        acquire_llm_slot()
        acquire_llm_slot()

    def test_local_limiter_rejects_zero(self) -> None:
        with self.assertRaises(ValueError):
            LLMRateLimiter(0)


if __name__ == "__main__":
    unittest.main()
