"""
Global LLM call rate limiting (requests per rolling 60-second window).

Shared across quick/deep models and all providers so concurrent graph nodes
still respect a single budget when configured.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from typing import Deque, Optional

_log = logging.getLogger(__name__)

_limiter_lock = threading.Lock()
_limiter: Optional["LLMRateLimiter"] = None


class LLMRateLimiter:
    """Sliding window: at most ``max_calls`` completions started in any 60s span."""

    __slots__ = ("_calls", "_lock", "max_calls")

    def __init__(self, max_calls: int) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be at least 1")
        self.max_calls = max_calls
        self._lock = threading.Lock()
        self._calls: Deque[float] = deque()

    def acquire(self) -> None:
        while True:
            wait = 0.0
            with self._lock:
                now = time.time()
                while self._calls and now - self._calls[0] >= 60.0:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(time.time())
                    return
                wait = 60.0 - (now - self._calls[0]) + 0.01
            wait = max(wait, 0.01)
            _log.debug("LLM rate limit: waiting %.3fs (%s/%s per 60s)", wait, len(self._calls), self.max_calls)
            time.sleep(wait)

    async def async_acquire(self) -> None:
        while True:
            wait = 0.0
            with self._lock:
                now = time.time()
                while self._calls and now - self._calls[0] >= 60.0:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(time.time())
                    return
                wait = 60.0 - (now - self._calls[0]) + 0.01
            wait = max(wait, 0.01)
            _log.debug("LLM rate limit: waiting %.3fs (async)", wait)
            await asyncio.sleep(wait)


def set_llm_rate_limit_rpm(rpm: Optional[float]) -> None:
    """
    Configure the process-wide LLM rate limit.

    Args:
        rpm: Max completed LLM requests per rolling 60 seconds, or None/<=0 to disable.
    """
    global _limiter
    with _limiter_lock:
        if rpm is None or rpm <= 0:
            _limiter = None
            _log.info("LLM rate limit disabled")
            return
        mc = max(1, int(rpm))
        _limiter = LLMRateLimiter(mc)
        _log.info("LLM rate limit enabled: %s requests per rolling 60s", mc)


def acquire_llm_slot() -> None:
    """Block until a slot is available (sync); no-op if limiting is off."""
    lim = _limiter
    if lim is not None:
        lim.acquire()


async def async_acquire_llm_slot() -> None:
    """Block until a slot is available (async); no-op if limiting is off."""
    lim = _limiter
    if lim is not None:
        await lim.async_acquire()
