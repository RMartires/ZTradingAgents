"""Extract Literal enum strings from ``final_state`` JSON for schedule CSV columns."""

from __future__ import annotations

import json
from typing import Any, Mapping

from pydantic import BaseModel

from tradingagents.backtest.dates_schedule import SCHEDULE_STRUCTURED_LITERAL_FIELDNAMES
from tradingagents.schemas import (
    AnalystReport,
    BullBearArgument,
    InvestmentPlanJudgment,
    RiskAnalystArgument,
    RiskAssessment,
    TradeProposal,
)

# Set in code (not by the LLM) when structured output failed and we used a plain-text recovery path.
STRUCTURED_JSON_RECOVERY_KEY = "_structured_error"


def model_dump_json_with_recovery(model: BaseModel, recovery: bool) -> str:
    """Serialize a structured model; if ``recovery``, tag JSON so CSV can show ``fallback``."""
    d = json.loads(model.model_dump_json())
    if recovery:
        d[STRUCTURED_JSON_RECOVERY_KEY] = True
    return json.dumps(d, ensure_ascii=False)


def _norm_buy_sell_hold(token: str) -> str:
    u = (token or "").strip().upper()
    return u if u in ("BUY", "SELL", "HOLD") else ""


def _parse(model: type, raw: Any):
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        d = json.loads(s)
        if isinstance(d, dict):
            d.pop(STRUCTURED_JSON_RECOVERY_KEY, None)
        return model.model_validate(d)
    except Exception:
        return None


def _parse_with_recovery(model: type, raw: Any) -> tuple[Any, bool]:
    """Like ``_parse`` but returns (model | None, recovery_flag)."""
    if raw is None:
        return None, False
    s = str(raw).strip()
    if not s:
        return None, False
    try:
        d = json.loads(s)
        recovery = False
        if isinstance(d, dict):
            recovery = bool(d.pop(STRUCTURED_JSON_RECOVERY_KEY, False))
        return model.model_validate(d), recovery
    except Exception:
        return None, False


def extract_structured_schedule_literals(
    final_state: Mapping[str, Any] | None,
) -> dict[str, str]:
    """
    Best-effort parse of structured JSON blobs on ``final_state``; only Literal scalars.

    Missing or invalid JSON yields empty strings for those cells.
    """
    empty = {k: "" for k in SCHEDULE_STRUCTURED_LITERAL_FIELDNAMES}
    if not final_state:
        return empty

    out = dict(empty)

    ar_m, rec_m = _parse_with_recovery(AnalystReport, final_state.get("market_report_structured"))
    if ar_m is not None:
        out["market_outlook"] = "fallback" if rec_m else ar_m.outlook

    ar_s, rec_s = _parse_with_recovery(AnalystReport, final_state.get("sentiment_report_structured"))
    if ar_s is not None:
        out["sentiment_outlook"] = "fallback" if rec_s else ar_s.outlook

    ar_n, rec_news = _parse_with_recovery(AnalystReport, final_state.get("news_report_structured"))
    if ar_n is not None:
        out["news_outlook"] = "fallback" if rec_news else ar_n.outlook

    ar_f, rec_f = _parse_with_recovery(AnalystReport, final_state.get("fundamentals_report_structured"))
    if ar_f is not None:
        out["fundamentals_outlook"] = "fallback" if rec_f else ar_f.outlook

    inv = final_state.get("investment_debate_state") or {}
    bb_bull, rec_bull = _parse_with_recovery(BullBearArgument, inv.get("bull_structured"))
    if bb_bull is not None:
        out["bull_implied_stance"] = "fallback" if rec_bull else bb_bull.implied_stance

    bb_bear, rec_bear = _parse_with_recovery(BullBearArgument, inv.get("bear_structured"))
    if bb_bear is not None:
        out["bear_implied_stance"] = "fallback" if rec_bear else bb_bear.implied_stance

    rds = final_state.get("risk_debate_state") or {}
    ra_a, rec_a = _parse_with_recovery(RiskAnalystArgument, rds.get("aggressive_structured"))
    if ra_a is not None:
        out["risk_aggressive_posture"] = "fallback" if rec_a else ra_a.risk_posture

    ra_c, rec_c = _parse_with_recovery(RiskAnalystArgument, rds.get("conservative_structured"))
    if ra_c is not None:
        out["risk_conservative_posture"] = "fallback" if rec_c else ra_c.risk_posture

    ra_n, rec_risk_neutral = _parse_with_recovery(RiskAnalystArgument, rds.get("neutral_structured"))
    if ra_n is not None:
        out["risk_neutral_posture"] = "fallback" if rec_risk_neutral else ra_n.risk_posture

    ip = _parse(InvestmentPlanJudgment, final_state.get("investment_plan_structured"))
    if ip is not None:
        out["investment_recommendation"] = _norm_buy_sell_hold(ip.recommendation)

    tp = _parse(TradeProposal, final_state.get("trader_investment_plan_structured"))
    if tp is not None:
        out["trader_decision"] = _norm_buy_sell_hold(tp.decision)

    ra = _parse(RiskAssessment, final_state.get("final_trade_decision_structured"))
    if ra is not None:
        out["risk_decision"] = _norm_buy_sell_hold(ra.decision)

    return out
