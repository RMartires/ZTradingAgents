import asyncio
import json
import logging
import os
import time
from contextvars import ContextVar
from typing import Any, Optional

import httpx
import openai

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI

from .base_client import BaseLLMClient
from .llm_rate_limit import (
    acquire_llm_slot,
    async_acquire_llm_slot,
    log_llm_completion_request,
)
from .validators import validate_model

_log = logging.getLogger(__name__)

_LAST_HTTP_RESPONSE: ContextVar[dict[str, Any] | None] = ContextVar(
    "_LAST_HTTP_RESPONSE", default=None
)


def _capture_http_response(resp: httpx.Response) -> None:
    try:
        body = resp.text
    except Exception:
        body = "<unavailable>"
    # Keep logs bounded; responses can be huge.
    if isinstance(body, str) and len(body) > 4000:
        body = body[:4000] + "\n…<truncated>…"
    _LAST_HTTP_RESPONSE.set(
        {
            "status_code": resp.status_code,
            "url": str(resp.url),
            "headers": dict(resp.headers),
            "body": body,
        }
    )


def _build_http_clients_with_capture() -> tuple[httpx.Client, httpx.AsyncClient]:
    hooks = {"response": [_capture_http_response]}
    return (
        httpx.Client(event_hooks=hooks),
        httpx.AsyncClient(event_hooks=hooks),
    )


def _should_capture_raw_response() -> bool:
    # Default on; can be disabled via env for noise/perf.
    v = os.getenv("LLM_CAPTURE_RAW_RESPONSE", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _provider_error_max_attempts() -> int:
    """How many times to retry when the API returns an error object inside a 200 JSON body (e.g. OpenRouter)."""
    try:
        n = int(os.getenv("LLM_PROVIDER_ERROR_MAX_ATTEMPTS", "4"))
    except ValueError:
        n = 4
    return max(1, min(n, 20))


def _retry_base_seconds() -> float:
    """Sleep before first retry (attempt 0). Env: ``LLM_RETRY_FIRST_WAIT_SEC`` (default 60)."""
    try:
        return max(0.0, float(os.getenv("LLM_RETRY_FIRST_WAIT_SEC", "60")))
    except ValueError:
        return 60.0


def _retry_step_seconds() -> float:
    """Extra seconds added per subsequent retry (per attempt index). Env: ``LLM_RETRY_STEP_SEC`` (default 30)."""
    try:
        return max(0.0, float(os.getenv("LLM_RETRY_STEP_SEC", "30")))
    except ValueError:
        return 30.0


def _is_retriable_openai_compatible_payload(err: object) -> bool:
    """True if LangChain raised ValueError(payload) from _create_chat_result and we should retry."""
    if not isinstance(err, dict):
        return False
    msg = err.get("message")
    if isinstance(msg, str):
        low = msg.lower()
        if any(
            s in low
            for s in (
                "rate limit",
                "too many requests",
                "timeout",
                "provider returned error",
                "temporarily",
                "overloaded",
            )
        ):
            return True
    code = err.get("code")
    if code is None:
        return False
    try:
        c = int(code)
    except (TypeError, ValueError):
        return False
    if c == 429:
        return True
    if 500 <= c <= 599:
        return True
    return c in (408, 409)


def _is_retriable_provider_value_error(exc: BaseException) -> bool:
    if not isinstance(exc, ValueError) or not exc.args:
        return False
    return _is_retriable_openai_compatible_payload(exc.args[0])


def _is_retriable_openai_sdk_error(exc: BaseException) -> bool:
    """True for HTTP/SDK failures that are worth retrying (429, 5xx, timeouts, connect errors)."""
    if isinstance(exc, openai.APIConnectionError):
        return True
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.InternalServerError):
        return True
    if isinstance(exc, openai.APIStatusError):
        code = exc.status_code
        if code in (408, 409, 429):
            return True
        if 500 <= code <= 599:
            return True
        return False
    return False


def _backoff_seconds(attempt: int) -> float:
    """Linear backoff: base + step * attempt (e.g. 60s, 90s, 120s, … with defaults).

    Used for all retriable failures (JSON-body errors, HTTP 429, 5xx, connection errors).
    """
    return _retry_base_seconds() + _retry_step_seconds() * float(attempt)


class UnifiedChatOpenAI(ChatOpenAI):
    """ChatOpenAI subclass that strips temperature/top_p for GPT-5 family models.

    GPT-5 family models use reasoning natively. temperature/top_p are only
    accepted when reasoning.effort is 'none'; with any other effort level
    (or for older GPT-5/GPT-5-mini/GPT-5-nano which always reason) the API
    rejects these params. Langchain defaults temperature=0.7, so we must
    strip it to avoid errors.

    Non-GPT-5 models (GPT-4.1, xAI, Ollama, etc.) are unaffected.

    Also retries when the HTTP response is 200 but the JSON body contains
    ``error`` (OpenRouter and some gateways report 502/429 that way), which
    bypasses the OpenAI SDK's HTTP-level retries.

    Retries ``openai.RateLimitError`` (HTTP 429), connection/timeouts, and 5xx
    class responses after the SDK's own ``max_retries`` are exhausted.

    Between attempts, waits ``LLM_RETRY_FIRST_WAIT_SEC + LLM_RETRY_STEP_SEC * attempt``
    seconds (defaults 60 + 30 per attempt), including for HTTP 429.
    """

    def __init__(self, **kwargs):
        if "gpt-5" in kwargs.get("model", "").lower():
            kwargs.pop("temperature", None)
            kwargs.pop("top_p", None)
        super().__init__(**kwargs)

    def _llm_log_label(self) -> str:
        mid = getattr(self, "model_name", None) or getattr(self, "model", None) or ""
        return f"ChatOpenAI model={mid}"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        attempts = _provider_error_max_attempts()
        last: Optional[BaseException] = None
        for attempt in range(attempts):
            try:
                acquire_llm_slot()
                log_llm_completion_request(self._llm_log_label())
                return super()._generate(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
            except json.JSONDecodeError as e:
                last = e
                if attempt == attempts - 1:
                    raise
                delay = _backoff_seconds(attempt)
                raw = _LAST_HTTP_RESPONSE.get()
                _log.warning(
                    "LLM response JSON decode error (attempt %s/%s), retrying in %.2fs: %s raw=%s",
                    attempt + 1,
                    attempts,
                    delay,
                    e,
                    raw,
                )
                time.sleep(delay)
            except ValueError as e:
                last = e
                if not _is_retriable_provider_value_error(e) or attempt == attempts - 1:
                    raise
                delay = _backoff_seconds(attempt)
                _log.warning(
                    "LLM provider error (attempt %s/%s), retrying in %.2fs: %s",
                    attempt + 1,
                    attempts,
                    delay,
                    e.args[0] if e.args else e,
                )
                time.sleep(delay)
            except openai.OpenAIError as e:
                last = e
                if not _is_retriable_openai_sdk_error(e) or attempt == attempts - 1:
                    raise
                delay = _backoff_seconds(attempt)
                _log.warning(
                    "LLM API error (attempt %s/%s), retrying in %.2fs: %s",
                    attempt + 1,
                    attempts,
                    delay,
                    e,
                )
                time.sleep(delay)
        assert last is not None
        raise last

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        attempts = _provider_error_max_attempts()
        last: Optional[BaseException] = None
        for attempt in range(attempts):
            try:
                await async_acquire_llm_slot()
                log_llm_completion_request(self._llm_log_label())
                return await super()._agenerate(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
            except json.JSONDecodeError as e:
                last = e
                if attempt == attempts - 1:
                    raise
                delay = _backoff_seconds(attempt)
                raw = _LAST_HTTP_RESPONSE.get()
                _log.warning(
                    "LLM response JSON decode error (attempt %s/%s), retrying in %.2fs: %s raw=%s",
                    attempt + 1,
                    attempts,
                    delay,
                    e,
                    raw,
                )
                await asyncio.sleep(delay)
            except ValueError as e:
                last = e
                if not _is_retriable_provider_value_error(e) or attempt == attempts - 1:
                    raise
                delay = _backoff_seconds(attempt)
                _log.warning(
                    "LLM provider error (attempt %s/%s), retrying in %.2fs: %s",
                    attempt + 1,
                    attempts,
                    delay,
                    e.args[0] if e.args else e,
                )
                await asyncio.sleep(delay)
            except openai.OpenAIError as e:
                last = e
                if not _is_retriable_openai_sdk_error(e) or attempt == attempts - 1:
                    raise
                delay = _backoff_seconds(attempt)
                _log.warning(
                    "LLM API error (attempt %s/%s), retrying in %.2fs: %s",
                    attempt + 1,
                    attempts,
                    delay,
                    e,
                )
                await asyncio.sleep(delay)
        assert last is not None
        raise last


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI, Ollama, OpenRouter, and xAI providers."""

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        provider: str = "openai",
        **kwargs,
    ):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()

    def get_llm(self) -> Any:
        """Return configured ChatOpenAI instance."""
        llm_kwargs = {"model": self.model}

        if self.provider == "xai":
            llm_kwargs["base_url"] = "https://api.x.ai/v1"
            api_key = os.environ.get("XAI_API_KEY")
            if api_key:
                llm_kwargs["api_key"] = api_key
        elif self.provider == "openrouter":
            llm_kwargs["base_url"] = "https://openrouter.ai/api/v1"
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if api_key:
                llm_kwargs["api_key"] = api_key
            if (
                "http_client" not in self.kwargs
                and "http_async_client" not in self.kwargs
                and _should_capture_raw_response()
            ):
                hc, ahc = _build_http_clients_with_capture()
                llm_kwargs["http_client"] = hc
                llm_kwargs["http_async_client"] = ahc
        elif self.provider == "ollama":
            llm_kwargs["base_url"] = "http://localhost:11434/v1"
            llm_kwargs["api_key"] = "ollama"  # Ollama doesn't require auth
        elif self.base_url:
            llm_kwargs["base_url"] = self.base_url

        for key in (
            "timeout",
            "max_retries",
            "max_tokens",
            "reasoning_effort",
            "api_key",
            "callbacks",
            "http_client",
            "http_async_client",
        ):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        return UnifiedChatOpenAI(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for the provider."""
        return validate_model(self.provider, self.model)
