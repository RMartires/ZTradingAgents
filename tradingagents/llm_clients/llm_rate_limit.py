"""
Global LLM call rate limiting (requests per rolling 60-second window).

Shared across quick/deep models and all providers so concurrent graph nodes
still respect a single budget when configured.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Deque, Optional, Tuple

_log = logging.getLogger(__name__)

_limiter_lock = threading.Lock()
_completion_log_configured = False
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

    def window_count(self) -> Tuple[int, int]:
        """(calls counted in the last 60s wall clock, max_calls cap). Does not mutate state."""
        with self._lock:
            now = time.time()
            active = sum(1 for t in self._calls if now - t < 60.0)
            return (active, self.max_calls)

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


def configure_llm_completion_logging() -> None:
    """
    Make per-call LLM completion lines visible when root ``LOG_LEVEL`` is WARNING (the default).

    Uses a dedicated stderr handler on this logger. Disable with ``LLM_COMPLETION_LOG=0``.
    """
    global _completion_log_configured
    if _completion_log_configured:
        return
    _completion_log_configured = True
    if os.getenv("LLM_COMPLETION_LOG", "1").strip().lower() in ("0", "false", "no", "off"):
        return
    _log.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    _log.addHandler(h)
    _log.propagate = False


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


def get_rate_limit_snapshot() -> Optional[Tuple[int, int]]:
    """If RPM limiting is on, return ``(calls_in_last_60s, cap)``; else ``None``."""
    lim = _limiter
    if lim is None:
        return None
    return lim.window_count()


def log_llm_completion_request(label: str = "") -> None:
    """
    Log one line per completion API attempt (after rate-limit slot, before HTTP).

    ``TradingAgentsGraph`` calls ``configure_llm_completion_logging()`` so these show even when
    ``LOG_LEVEL=WARNING``. Set ``LLM_COMPLETION_LOG=0`` to turn them off.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    tag = label.strip() or "request"
    snap = get_rate_limit_snapshot()
    if snap is not None:
        used, cap = snap
        _log.info(
            "LLM completion [%s] at %s rate_window=%s/%s (rolling 60s)",
            tag,
            ts,
            used,
            cap,
        )
    else:
        _log.info(
            "LLM completion [%s] at %s (rate_limit off)",
            tag,
            ts,
        )
