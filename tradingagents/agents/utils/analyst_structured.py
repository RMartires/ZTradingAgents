"""Second-pass structured extraction from analyst prose (uses ``schemas.outputs`` JSON-schema path only)."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from tradingagents.backtest.structured_literals import model_dump_json_with_recovery
from tradingagents.llm_clients.invoke_fallback import (
    STRUCTURED_OUTPUT_PROMPT_PREFIX,
    format_llm_response_for_log,
    log_structured_parse_failure,
    make_structured_runnable,
    resolved_structured_output_method,
    structured_output_needs_content_leadin,
)
from tradingagents.schemas import AnalystReport, structured_prompt_example_suffix

_log = logging.getLogger(__name__)


def _escape_langchain_template_braces(s: str) -> str:
    """Double ``{`` / ``}`` so JSON in examples is literal for :class:`ChatPromptTemplate` (renders as single braces)."""
    return s.replace("{", "{{").replace("}", "}}")


def _fallback_analyst_report(draft: str) -> tuple[AnalystReport, bool]:
    """When structured extraction fails, keep prose in ``report``; CSV ``fallback`` via recovery flag."""
    text = (draft or "").strip()
    return (
        AnalystReport(report=text, headline="", key_findings=[], outlook="mixed"),
        True,
    )


def analyst_report_from_draft(llm: Any, draft: str) -> tuple[AnalystReport, bool]:
    """Map free-form analyst output into AnalystReport. Second value is True if error-recovery path."""
    if not (draft or "").strip():
        return AnalystReport(report="", headline="", key_findings=[], outlook="mixed"), False

    method = resolved_structured_output_method()
    body = """Put the full report text in `report`
(may be the same as the draft). Summarize `headline` and bullet `key_findings`. Classify `outlook`
as one of: bullish, bearish, mixed, neutral.

--- DRAFT ---
{draft}
---"""
    body = body + _escape_langchain_template_braces(
        structured_prompt_example_suffix(AnalystReport)
    )
    tmpl = ChatPromptTemplate.from_template(
        (STRUCTURED_OUTPUT_PROMPT_PREFIX + body)
        if structured_output_needs_content_leadin(method)
        else body
    )
    chain = tmpl | make_structured_runnable(llm, AnalystReport, method)
    try:
        out = chain.invoke({"draft": draft})
        if isinstance(out, dict):
            err = out.get("parsing_error")
            raw = out.get("raw")
            if err is not None:
                log_structured_parse_failure("AnalystReport extraction", err, raw)
                return _fallback_analyst_report(draft)
            parsed = out.get("parsed")
            if parsed is not None:
                return parsed, False
            _log.warning(
                "AnalystReport structured output missing parsed; raw=%s",
                format_llm_response_for_log(raw),
            )
            return _fallback_analyst_report(draft)
        return out, False
    except Exception as exc:
        _log.warning(
            "AnalystReport structured extraction failed (%s: %s); using recovery outlook.",
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        return _fallback_analyst_report(draft)


def analyst_report_json_for_state(llm: Any, draft: str) -> str:
    """JSON blob for graph state (includes ``_structured_error`` when extraction failed)."""
    model, recovery = analyst_report_from_draft(llm, draft)
    return model_dump_json_with_recovery(model, recovery)
