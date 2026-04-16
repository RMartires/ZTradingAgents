"""Optional fallback invoke when the primary (e.g. deep) model fails after retries."""

from __future__ import annotations

import copy
import logging
import os
from typing import Any, Callable, Optional, Tuple, Type, TypeVar, cast

import openai
from pydantic import BaseModel

from tradingagents.llm_clients.openai_client import _is_retriable_provider_value_error
from tradingagents.schemas.outputs import (
    DEFAULT_STRUCTURED_LLM_METHOD,
    structured_prompt_example_suffix,
)

_log = logging.getLogger(__name__)

# Some OpenRouter models (e.g. Qwen) expect this lead-in so the assistant returns parseable JSON
# in ``message.content`` when using ``response_format`` / JSON-schema structured output (not tool calls).
STRUCTURED_OUTPUT_PROMPT_PREFIX = (
    "Extract structured fields from this analyst draft into a JSON object.\n\n"
)


def structured_output_needs_content_leadin(method: str) -> bool:
    """True when structured output is steered via JSON in message content (``json_schema`` / ``json_mode``)."""
    m = (method or DEFAULT_STRUCTURED_LLM_METHOD).strip().lower()
    return m in ("json_schema", "json_mode")


def resolved_structured_output_method(explicit: Optional[str] = None) -> str:
    """Resolve structured method for :mod:`tradingagents.schemas.outputs` pipeline models only.

    Allowed: ``json_schema``, ``json_mode`` (see :data:`DEFAULT_STRUCTURED_LLM_METHOD`).
    ``function_calling`` is ignored—we do not use tool-calling for these schemas.
    Override via env ``LLM_STRUCTURED_OUTPUT_METHOD`` or ``explicit``.
    """
    if explicit is not None:
        m = explicit.strip().lower()
    else:
        m = (os.getenv("LLM_STRUCTURED_OUTPUT_METHOD") or DEFAULT_STRUCTURED_LLM_METHOD).strip().lower()
    if m in ("json_schema", "json_mode"):
        return m
    if m == "function_calling":
        _log.warning(
            "LLM_STRUCTURED_OUTPUT_METHOD=function_calling is not used for "
            "tradingagents.schemas.outputs; using %s.",
            DEFAULT_STRUCTURED_LLM_METHOD,
        )
    return DEFAULT_STRUCTURED_LLM_METHOD


def ensure_structured_output_prompt_prefix(text: str) -> str:
    """Prepend :data:`STRUCTURED_OUTPUT_PROMPT_PREFIX` once (idempotent)."""
    s = text or ""
    head = STRUCTURED_OUTPUT_PROMPT_PREFIX.strip()
    if s.lstrip().startswith(head):
        return s
    return STRUCTURED_OUTPUT_PROMPT_PREFIX + s


def ensure_structured_output_messages_prefix(messages: Any) -> Any:
    """Prepend the structured-output prefix to the first user/human message content."""
    if not isinstance(messages, list) or not messages:
        return messages
    out: list[Any] = []
    prefixed = False
    for m in messages:
        if isinstance(m, dict):
            d = dict(m)
            if (
                not prefixed
                and d.get("role") == "user"
                and isinstance(d.get("content"), str)
            ):
                d["content"] = ensure_structured_output_prompt_prefix(d["content"])
                prefixed = True
            out.append(d)
            continue
        # LangChain BaseMessage: first human message
        t = getattr(m, "type", None)
        if not prefixed and t == "human":
            content = getattr(m, "content", "") or ""
            try:
                out.append(
                    m.model_copy(  # type: ignore[union-attr]
                        update={"content": ensure_structured_output_prompt_prefix(str(content))}
                    )
                )
            except AttributeError:
                out.append(m)
            prefixed = True
            continue
        out.append(m)
    return out


def _append_structured_example_to_messages(messages: Any, schema: Type[BaseModel]) -> Any:
    """Append schema-specific JSON key example to the first user/human message (trader, etc.)."""
    suffix = structured_prompt_example_suffix(schema)
    if not suffix or not isinstance(messages, list):
        return messages
    out: list[Any] = []
    appended = False
    for m in messages:
        if appended:
            out.append(m)
            continue
        if isinstance(m, dict):
            d = dict(m)
            if d.get("role") == "user" and isinstance(d.get("content"), str):
                d["content"] = d["content"] + suffix
                out.append(d)
                appended = True
                continue
        t = getattr(m, "type", None)
        if t == "human":
            content = getattr(m, "content", "") or ""
            try:
                out.append(
                    m.model_copy(  # type: ignore[union-attr]
                        update={"content": str(content) + suffix}
                    )
                )
            except AttributeError:
                out.append(m)
            appended = True
            continue
        out.append(m)
    return out


class StructuredParseError(Exception):
    """Structured output reported ``parsing_error`` (already logged with raw message)."""


def format_llm_response_for_log(msg: Any, *, max_chars: int = 16_000) -> str:
    """Serialize an AIMessage (or similar) for debug logs; truncates very long payloads."""
    if msg is None:
        return "(raw LLM message is None)"
    chunks: list[str] = []
    for name in ("content", "tool_calls", "additional_kwargs", "response_metadata", "id"):
        if hasattr(msg, name):
            chunks.append(f"{name}={getattr(msg, name)!r}")
    if not chunks:
        return repr(msg)[:max_chars]
    out = "\n".join(chunks)
    if len(out) > max_chars:
        return out[:max_chars] + f"\n... [truncated, {len(out)} chars total]"
    return out


def log_structured_parse_failure(context: str, err: BaseException, raw: Any) -> None:
    """Log full traceback and raw model message when ``with_structured_output(..., include_raw=True)`` reports a parse error."""
    _log.warning(
        "%s: structured output parsing failed; raw LLM message:\n%s",
        context,
        format_llm_response_for_log(raw),
    )
    _log.warning(
        "%s: parsing exception (full traceback follows):",
        context,
        exc_info=(type(err), err, err.__traceback__),
    )


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
            exc_info=True,
        )
        return fallback_llm.invoke(prompt)


T = TypeVar("T", bound=BaseModel)


def _structured_strict_enabled() -> bool:
    """Env ``LLM_STRUCTURED_STRICT`` (default true): pass ``strict=True`` to structured output API."""
    v = (os.getenv("LLM_STRUCTURED_STRICT") or "true").strip().lower()
    return v not in ("0", "false", "no", "off")


def _llm_model_id_lower(llm: Any) -> str:
    """Best-effort model id for provider quirks (e.g. GPT-5 rejects temperature on some routes)."""
    mid = getattr(llm, "model_name", None) or getattr(llm, "model", None)
    if mid is not None and str(mid).strip():
        return str(mid).lower()
    return ""


def _structured_temperature_from_env(llm: Any) -> Optional[float]:
    """Env ``LLM_STRUCTURED_TEMPERATURE``: sampling temp for :mod:`tradingagents.schemas.outputs` invokes only.

    If unset or empty, returns ``None`` (leave the base model's default temperature).
    GPT-5 family models often reject ``temperature``; we skip binding in that case.
    """
    raw = os.getenv("LLM_STRUCTURED_TEMPERATURE")
    if raw is None or not str(raw).strip():
        return None
    try:
        t = float(str(raw).strip())
    except (TypeError, ValueError):
        _log.warning(
            "LLM_STRUCTURED_TEMPERATURE=%r is not a float; ignoring structured temperature.",
            raw,
        )
        return None
    model_id = _llm_model_id_lower(llm)
    if "gpt-5" in model_id:
        _log.warning(
            "LLM_STRUCTURED_TEMPERATURE is set but model id %r may reject temperature; "
            "skipping structured temperature bind.",
            model_id or "(unknown)",
        )
        return None
    return t


def bound_llm_for_structured_output(llm: Any) -> Any:
    """Bind params used only for ``schemas.outputs`` structured JSON invokes.

    - ``LLM_STRUCTURED_MAX_TOKENS`` (default 16384) as ``max_tokens``.
    - ``LLM_STRUCTURED_TEMPERATURE`` (optional): overrides sampling for these calls only;
      if unset, the graph LLM's default temperature is unchanged for structured passes too.
    """
    bind_kw: dict[str, Any] = {}
    try:
        max_t = int(os.getenv("LLM_STRUCTURED_MAX_TOKENS", "16384"))
        if max_t > 0:
            bind_kw["max_tokens"] = max_t
    except (TypeError, ValueError):
        pass
    st = _structured_temperature_from_env(llm)
    if st is not None:
        bind_kw["temperature"] = st
    if not bind_kw:
        return llm
    return llm.bind(**bind_kw)


def make_structured_runnable(llm: Any, schema: Type[Any], structured_method: str) -> Any:
    """``bound_llm | with_structured_output(..., strict=…)`` for pipeline schemas."""
    bound = bound_llm_for_structured_output(llm)
    kw: dict[str, Any] = {
        "include_raw": True,
        "method": structured_method,
    }
    if _structured_strict_enabled():
        kw["strict"] = True
    return bound.with_structured_output(schema, **kw)


def _structured_invoke_parsed(
    llm: Any,
    invoke_input: Any,
    schema: Type[T],
    context: str,
    *,
    messages: bool,
    structured_method: str,
) -> T:
    """
    Invoke ``with_structured_output(..., include_raw=True)`` and return parsed model,
    or raise :class:`StructuredParseError` after logging raw message + traceback.
    """
    _ = messages  # reserved if we need different branches later
    structured = make_structured_runnable(llm, schema, structured_method)
    out = structured.invoke(invoke_input)
    if not isinstance(out, dict):
        return cast(T, out)
    err = out.get("parsing_error")
    raw = out.get("raw")
    parsed = out.get("parsed")
    if err is not None:
        log_structured_parse_failure(context, err, raw)
        raise StructuredParseError(str(err)) from err
    if parsed is None:
        _log.warning(
            "%s: structured output missing parsed result (unexpected); raw=%s",
            context,
            format_llm_response_for_log(raw),
        )
        raise StructuredParseError("missing parsed") from None
    return cast(T, parsed)


def invoke_structured_with_deep_fallback(
    primary: Any,
    prompt: str,
    schema: Type[T],
    *,
    fallback_llm: Optional[Any] = None,
    context: str = "structured LLM",
    structured_method: Optional[str] = None,
) -> T:
    """
    ``primary.with_structured_output(schema).invoke(prompt)``, with the same
    fallback policy as ``invoke_chat_with_deep_fallback`` when the primary fails.
    Uses ``include_raw=True`` so parse failures log the raw AIMessage.

    ``structured_method`` defaults to :func:`resolved_structured_output_method` (env
    ``LLM_STRUCTURED_OUTPUT_METHOD``, default :data:`~tradingagents.schemas.outputs.DEFAULT_STRUCTURED_LLM_METHOD`).
    """
    method = resolved_structured_output_method(structured_method)
    try:
        return _structured_invoke_parsed(
            primary, prompt, schema, context, messages=False, structured_method=method
        )
    except StructuredParseError:
        raise
    except Exception as exc:
        if fallback_llm is None or not _deep_failure_allows_quick_fallback(exc):
            raise
        _log.warning(
            "%s: primary structured invoke failed (%s); retrying once with fallback: %s",
            context,
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        return _structured_invoke_parsed(
            fallback_llm,
            prompt,
            schema,
            context,
            messages=False,
            structured_method=method,
        )


def invoke_structured_messages_with_deep_fallback(
    primary: Any,
    messages: Any,
    schema: Type[T],
    *,
    fallback_llm: Optional[Any] = None,
    context: str = "structured LLM",
    structured_method: Optional[str] = None,
) -> T:
    """Structured output from a message list (e.g. trader system+user)."""
    method = resolved_structured_output_method(structured_method)
    try:
        return _structured_invoke_parsed(
            primary, messages, schema, context, messages=True, structured_method=method
        )
    except StructuredParseError:
        raise
    except Exception as exc:
        if fallback_llm is None or not _deep_failure_allows_quick_fallback(exc):
            raise
        _log.warning(
            "%s: primary structured invoke failed (%s); retrying once with fallback: %s",
            context,
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        return _structured_invoke_parsed(
            fallback_llm,
            messages,
            schema,
            context,
            messages=True,
            structured_method=method,
        )


def _plain_invoke_text(result: Any) -> str:
    if result is None:
        return ""
    content = getattr(result, "content", None)
    if content is not None:
        return str(content)
    return str(result)


def invoke_structured_prompt_or_plain(
    primary: Any,
    prompt: str,
    schema: Type[T],
    *,
    build_from_text: Callable[[str], T],
    fallback_llm: Optional[Any] = None,
    context: str = "structured LLM",
    structured_method: Optional[str] = None,
) -> Tuple[T, bool]:
    """
    Try structured output (with deep fallback for rate limits); on any failure,
    plain ``invoke(prompt)`` and map text through ``build_from_text`` so the graph continues.

    Returns ``(model, used_plain_text_fallback)``. The second flag is True only when
    structured output failed and the plain path was used (for ``_structured_error`` in JSON).

    The Qwen/OpenRouter JSON lead-in is applied only when ``structured_method`` resolves to
    ``json_schema`` or ``json_mode`` (see :func:`structured_output_needs_content_leadin`).
    """
    method = resolved_structured_output_method(structured_method)
    prompt_f = (
        ensure_structured_output_prompt_prefix(prompt)
        if structured_output_needs_content_leadin(method)
        else prompt
    )
    prompt_f = prompt_f + structured_prompt_example_suffix(schema)
    try:
        return (
            invoke_structured_with_deep_fallback(
                primary,
                prompt_f,
                schema,
                fallback_llm=fallback_llm,
                context=context,
                structured_method=method,
            ),
            False,
        )
    except StructuredParseError:
        # Raw message + traceback already logged in log_structured_parse_failure.
        pass
    except Exception as exc:
        _log.warning(
            "%s: structured invoke failed (%s: %s); using plain text completion.",
            context,
            type(exc).__name__,
            exc,
            exc_info=True,
        )
    llm = fallback_llm or primary
    # Plain completion is not JSON-schema structured; use the original prompt (no Qwen JSON prefix).
    raw = llm.invoke(prompt)
    return build_from_text(_plain_invoke_text(raw)), True


def invoke_structured_messages_or_plain(
    primary: Any,
    messages: Any,
    schema: Type[T],
    *,
    build_from_text: Callable[[str], T],
    fallback_llm: Optional[Any] = None,
    context: str = "structured LLM",
    structured_method: Optional[str] = None,
) -> Tuple[T, bool]:
    """Same as :func:`invoke_structured_prompt_or_plain` for message lists."""
    method = resolved_structured_output_method(structured_method)
    base = copy.deepcopy(messages) if messages is not None else messages
    messages_f = (
        ensure_structured_output_messages_prefix(base)
        if structured_output_needs_content_leadin(method)
        else base
    )
    messages_f = _append_structured_example_to_messages(messages_f, schema)
    try:
        return (
            invoke_structured_messages_with_deep_fallback(
                primary,
                messages_f,
                schema,
                fallback_llm=fallback_llm,
                context=context,
                structured_method=method,
            ),
            False,
        )
    except StructuredParseError:
        pass
    except Exception as exc:
        _log.warning(
            "%s: structured invoke failed (%s: %s); using plain text completion.",
            context,
            type(exc).__name__,
            exc,
            exc_info=True,
        )
    llm = fallback_llm or primary
    raw = llm.invoke(messages)
    return build_from_text(_plain_invoke_text(raw)), True
