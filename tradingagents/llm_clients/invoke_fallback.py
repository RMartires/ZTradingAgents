"""Optional fallback invoke when the primary (e.g. deep) model fails after retries."""

from __future__ import annotations

import logging
from typing import Any, Optional

import openai

from tradingagents.llm_clients.openai_client import _is_retriable_provider_value_error

_log = logging.getLogger(__name__)


def _deep_failure_allows_quick_fallback(exc: BaseException) -> bool:
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIConnectionError):
        return True
    if isinstance(exc, openai.InternalServerError):
        return True
    if isinstance(exc, openai.APIStatusError):
        code = getattr(exc, "status_code", None)
        if code is not None and (code == 429 or 500 <= code <= 599):
            return True
        return False
    if isinstance(exc, ValueError) and _is_retriable_provider_value_error(exc):
        return True
    return False


def invoke_chat_with_deep_fallback(
    primary: Any,
    prompt: str,
    *,
    fallback_llm: Optional[Any] = None,
    context: str = "LLM node",
) -> Any:
    """
    ``primary.invoke(prompt)``, or if that raises after the client's own retries,
    ``fallback_llm.invoke(prompt)`` once when the error looks like overload / 429 / 5xx.
    """
    try:
        return primary.invoke(prompt)
    except Exception as exc:
        if fallback_llm is None or not _deep_failure_allows_quick_fallback(exc):
            raise
        _log.warning(
            "%s: primary model failed (%s); retrying once with fallback model: %s",
            context,
            type(exc).__name__,
            exc,
        )
        return fallback_llm.invoke(prompt)
