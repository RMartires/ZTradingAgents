"""Pydantic models for structured LLM outputs at each pipeline stage."""

from __future__ import annotations

from typing import Annotated, Literal, Type

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

# Pipeline structured completions use LangChain ``with_structured_output(..., method=…)``
# with **JSON in message content** (``json_schema`` or ``json_mode``). We do **not** use the
# tool-calling structured path (``function_calling`` / ``tool_choice``), which breaks on some
# OpenRouter routes and is unrelated to these schemas.
#
# Optional env ``LLM_STRUCTURED_TEMPERATURE``: sampling temperature only for these schema
# invokes (see ``invoke_fallback.bound_llm_for_structured_output``), not plain-text steps.
DEFAULT_STRUCTURED_LLM_METHOD: Literal["json_schema", "json_mode"] = "json_schema"

# Caps keep structured JSON completions small so Literal fields fit under token limits.
# Long-form prose lives in the preceding non-structured turns; this pass is for enums + short text.
_ANALYSIS_MAX = 2500
_NARRATIVE_MAX = 1800
_RATIONALE_MAX = 600
_STRATEGIC_MAX = 800
_REPORT_MAX = 6000
_HEADLINE_MAX = 240
_FINDING_ITEM_MAX = 280
_MAX_FINDINGS = 10


def _truncate_str(max_len: int):
    """Coerce input to str and truncate so validation never fails on length (incl. plain-text fallback)."""

    def _inner(v: object) -> str:
        if v is None:
            return ""
        s = str(v)
        if len(s) <= max_len:
            return s
        return s[: max_len - 3].rstrip() + "..."

    return _inner


def _truncate_findings_list(v: object) -> list[str]:
    if v is None:
        return []
    if not isinstance(v, list):
        return []
    out: list[str] = []
    for item in v[:_MAX_FINDINGS]:
        out.append(_truncate_str(_FINDING_ITEM_MAX)(item))
    return out


ShortAnalysis = Annotated[str, BeforeValidator(_truncate_str(_ANALYSIS_MAX))]
ShortNarrative = Annotated[str, BeforeValidator(_truncate_str(_NARRATIVE_MAX))]
ShortRationale = Annotated[str, BeforeValidator(_truncate_str(_RATIONALE_MAX))]
ShortStrategic = Annotated[str, BeforeValidator(_truncate_str(_STRATEGIC_MAX))]
ShortReport = Annotated[str, BeforeValidator(_truncate_str(_REPORT_MAX))]
ShortHeadline = Annotated[str, BeforeValidator(_truncate_str(_HEADLINE_MAX))]
FindingsList = Annotated[list[str], BeforeValidator(_truncate_findings_list)]

# Bull/bear debate stance is trading intent (buy/sell/hold/neutral). Models often emit analyst-style
# outlook words (bullish/bearish) from context; coerce before Literal validation.
_STANCE_ALLOWED = frozenset({"buy", "sell", "hold", "neutral"})


def _coerce_implied_stance(v: object) -> str:
    if v is None:
        return "neutral"
    s = str(v).strip().lower()
    synonym = {"bullish": "buy", "bearish": "sell", "mixed": "neutral"}
    s = synonym.get(s, s)
    if s in _STANCE_ALLOWED:
        return s
    return "neutral"


ImpliedStance = Annotated[
    Literal["buy", "sell", "hold", "neutral"],
    BeforeValidator(_coerce_implied_stance),
]

# Risk debators are named Aggressive / Conservative / Neutral; models often emit those labels
# instead of the schema literals high / moderate / low.
_RISK_POSTURE_ALLOWED = frozenset({"high", "moderate", "low"})


def _coerce_risk_posture(v: object) -> str:
    if v is None:
        return "moderate"
    s = str(v).strip().lower()
    synonym = {"aggressive": "high", "conservative": "low", "neutral": "moderate"}
    s = synonym.get(s, s)
    if s in _RISK_POSTURE_ALLOWED:
        return s
    return "moderate"


RiskPosture = Annotated[
    Literal["high", "moderate", "low"],
    BeforeValidator(_coerce_risk_posture),
]


def _decision_token_for_final_tx_line(decision: object) -> str:
    """Normalize Buy/Sell/Hold or BUY/SELL/HOLD to uppercase token for FINAL TRANSACTION PROPOSAL."""
    if decision is None:
        return "HOLD"
    s = str(decision).strip()
    u = s.upper()
    if u in ("BUY", "SELL", "HOLD"):
        return u
    low = s.lower()
    if low == "buy":
        return "BUY"
    if low == "sell":
        return "SELL"
    return "HOLD"


def _backfill_missing_narrative(data: object) -> object:
    """If structured JSON omits ``narrative``, use ``rationale`` or a minimal FINAL line (flaky models)."""
    if not isinstance(data, dict):
        return data
    n = data.get("narrative")
    if n is not None and isinstance(n, str) and n.strip():
        return data
    out = dict(data)
    r = out.get("rationale")
    if isinstance(r, str) and r.strip():
        out["narrative"] = r
        return out
    tok = _decision_token_for_final_tx_line(out.get("decision"))
    out["narrative"] = f"FINAL TRANSACTION PROPOSAL: **{tok}**"
    return out


class AnalystReport(BaseModel):
    """Structured extraction from an analyst's prose report. Literal outlook first for schema emphasis."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "outlook": "bullish",
                    "headline": "Lead summary",
                    "key_findings": ["Finding one", "Finding two"],
                    "report": "Full prose here.",
                }
            ]
        }
    )

    outlook: Literal["bullish", "bearish", "mixed", "neutral"] = Field(
        default="mixed",
        description="Required: one of bullish, bearish, mixed, neutral.",
    )
    headline: ShortHeadline = Field(default="", description="Short headline (max ~240 chars).")
    key_findings: FindingsList = Field(
        default_factory=list,
        description=f"Up to {_MAX_FINDINGS} bullets; each line max ~{_FINDING_ITEM_MAX} chars.",
    )
    report: ShortReport = Field(
        default="",
        description="Full report text; keep concise (max ~6000 chars). Root keys: outlook, headline, key_findings, report.",
    )


class BullBearArgument(BaseModel):
    """Bull or bear debate: stance (Literal) first, then short analysis."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "implied_stance": "hold",
                    "analysis": "Short conversational argument text.",
                }
            ]
        }
    )

    implied_stance: ImpliedStance = Field(
        default="neutral",
        description=(
            "Trading stance: buy, sell, hold, or neutral — not bullish/bearish/mixed "
            "(those are analyst outlook labels elsewhere)."
        ),
    )
    analysis: ShortAnalysis = Field(
        description='Required. Conversational argument. JSON keys must be exactly "implied_stance" and "analysis" only.',
    )


class RiskAnalystArgument(BaseModel):
    """Aggressive / neutral / conservative risk debate turn."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "risk_posture": "moderate",
                    "analysis": "Short risk-focused argument text.",
                }
            ]
        }
    )

    risk_posture: RiskPosture = Field(
        default="moderate",
        description="Required: one of high, moderate, low (not role names like Aggressive).",
    )
    analysis: ShortAnalysis = Field(
        description='Required. JSON keys must be exactly "risk_posture" and "analysis" only.',
    )


class InvestmentPlanJudgment(BaseModel):
    """Research manager: recommendation first, then short text fields."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "recommendation": "Hold",
                    "rationale": "Why this stance.",
                    "strategic_actions": "Concrete next steps.",
                    "narrative": "Full conversational plan for the trader.",
                }
            ]
        }
    )

    recommendation: Literal["Buy", "Sell", "Hold"] = Field(
        description='Required: exactly "Buy", "Sell", or "Hold" (title case).',
    )
    rationale: ShortRationale = Field(
        default="",
        description="Why this recommendation (max ~600 chars).",
    )
    strategic_actions: ShortStrategic = Field(
        default="",
        description="Concrete steps (max ~800 chars).",
    )
    narrative: ShortNarrative = Field(
        description=(
            "Required. Concise plan for the trader (max ~1800 chars). "
            "Keys: recommendation, rationale, strategic_actions, narrative — no aliases like decision/summary."
        ),
    )


class TradeProposal(BaseModel):
    """Trader: decision (Literal) first."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "decision": "HOLD",
                    "rationale": "Brief reason.",
                    "narrative": "Analysis text.\n\nFINAL TRANSACTION PROPOSAL: **HOLD**",
                }
            ]
        }
    )

    decision: Literal["BUY", "SELL", "HOLD"] = Field(
        description='Required: BUY, SELL, or HOLD. Root keys: decision, rationale, narrative.',
    )
    rationale: ShortRationale = Field(default="", description="Brief rationale (max ~600 chars).")
    narrative: ShortNarrative = Field(
        description="Required. Must end with FINAL TRANSACTION PROPOSAL line matching decision; max ~1800 chars.",
    )

    @model_validator(mode="before")
    @classmethod
    def _narrative_from_rationale_if_missing(cls, data: object) -> object:
        return _backfill_missing_narrative(data)


class RiskAssessment(BaseModel):
    """Risk manager: final recommendation."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "decision": "Hold",
                    "rationale": "Brief reason.",
                    "narrative": "Judge summary.\n\nFINAL TRANSACTION PROPOSAL: **HOLD**",
                }
            ]
        }
    )

    decision: Literal["Buy", "Sell", "Hold"] = Field(
        description='Required: Buy, Sell, or Hold. Root keys: decision, rationale, narrative.',
    )
    rationale: ShortRationale = Field(default="", description="Brief rationale (max ~600 chars).")
    narrative: ShortNarrative = Field(
        description="Required. Judge response; include FINAL TRANSACTION PROPOSAL; max ~1800 chars.",
    )

    @model_validator(mode="before")
    @classmethod
    def _narrative_from_rationale_if_missing(cls, data: object) -> object:
        return _backfill_missing_narrative(data)


# One-line hints appended to user prompts for structured extraction (exact keys for flaky models).
STRUCTURED_PROMPT_EXAMPLES: dict[str, str] = {
    "AnalystReport": (
        'Example JSON (use only these keys, same spelling): '
        '{"outlook":"mixed","headline":"Title","key_findings":["A","B"],"report":"Prose here."}'
    ),
    "BullBearArgument": (
        "implied_stance must be buy|sell|hold|neutral only (not bullish/bearish/mixed). "
        'Example: {"implied_stance":"buy","analysis":"Your argument."}'
    ),
    "RiskAnalystArgument": (
        'Example JSON (only keys risk_posture and analysis): '
        '{"risk_posture":"moderate","analysis":"Your argument."}'
    ),
    "InvestmentPlanJudgment": (
        'Example JSON (keys recommendation, rationale, strategic_actions, narrative): '
        '{"recommendation":"Hold","rationale":"Why.","strategic_actions":"Steps.","narrative":"Plan."}'
    ),
    "TradeProposal": (
        'Example JSON (keys decision, rationale, narrative; decision is BUY|SELL|HOLD): '
        '{"decision":"HOLD","rationale":"Why.","narrative":"Text ending with FINAL TRANSACTION PROPOSAL: **HOLD**"}'
    ),
    "RiskAssessment": (
        'Example JSON (keys decision, rationale, narrative; decision is Buy|Sell|Hold): '
        '{"decision":"Hold","rationale":"Why.","narrative":"Text ending with FINAL TRANSACTION PROPOSAL: **HOLD**"}'
    ),
}


def structured_prompt_example_suffix(schema: Type[BaseModel]) -> str:
    """Append to structured prompts so models see valid key names (OpenRouter/Qwen)."""
    hint = STRUCTURED_PROMPT_EXAMPLES.get(schema.__name__)
    if not hint:
        return ""
    return f"\n\n{hint}"


def append_final_tx_line_if_missing(narrative: str, token: str) -> str:
    """Ensure narrative contains FINAL TRANSACTION PROPOSAL for downstream signal parsing."""
    text = (narrative or "").strip()
    upper = text.upper()
    if "FINAL TRANSACTION PROPOSAL" in upper:
        return text
    t = token.strip().upper()
    if not t:
        return text
    suffix = f"\n\nFINAL TRANSACTION PROPOSAL: **{t}**"
    return text + suffix if text else suffix.strip()
