"""Tests for structured Literal extraction into schedule columns."""

import unittest

from tradingagents.backtest.dates_schedule import SCHEDULE_STRUCTURED_LITERAL_FIELDNAMES
from tradingagents.backtest.structured_literals import (
    extract_structured_schedule_literals,
    model_dump_json_with_recovery,
)
from tradingagents.schemas import (
    AnalystReport,
    BullBearArgument,
    InvestmentPlanJudgment,
    RiskAnalystArgument,
    RiskAssessment,
    TradeProposal,
)


class TestSchemaCoercion(unittest.TestCase):
    def test_risk_posture_synonyms(self):
        a = RiskAnalystArgument.model_validate(
            {"risk_posture": "Aggressive", "analysis": "x"}
        )
        self.assertEqual(a.risk_posture, "high")
        c = RiskAnalystArgument.model_validate(
            {"risk_posture": "conservative", "analysis": "x"}
        )
        self.assertEqual(c.risk_posture, "low")

    def test_risk_assessment_narrative_backfill(self):
        ra = RiskAssessment.model_validate(
            {
                "decision": "Hold",
                "rationale": "Only rationale, no narrative key.",
            }
        )
        self.assertEqual(ra.narrative, "Only rationale, no narrative key.")

    def test_trade_proposal_narrative_backfill(self):
        tp = TradeProposal.model_validate(
            {
                "decision": "HOLD",
                "rationale": "Rationale only.",
            }
        )
        self.assertEqual(tp.narrative, "Rationale only.")


class TestExtractStructuredScheduleLiterals(unittest.TestCase):
    def test_empty_state(self):
        out = extract_structured_schedule_literals(None)
        for k in SCHEDULE_STRUCTURED_LITERAL_FIELDNAMES:
            self.assertEqual(out.get(k), "", k)

    def test_full_state(self):
        final_state = {
            "market_report_structured": AnalystReport(
                report="x", outlook="bullish"
            ).model_dump_json(),
            "sentiment_report_structured": AnalystReport(
                report="x", outlook="neutral"
            ).model_dump_json(),
            "news_report_structured": AnalystReport(
                report="x", outlook="mixed"
            ).model_dump_json(),
            "fundamentals_report_structured": AnalystReport(
                report="x", outlook="bearish"
            ).model_dump_json(),
            "investment_debate_state": {
                "bull_structured": BullBearArgument(
                    analysis="a", implied_stance="buy"
                ).model_dump_json(),
                "bear_structured": BullBearArgument(
                    analysis="b", implied_stance="sell"
                ).model_dump_json(),
            },
            "risk_debate_state": {
                "aggressive_structured": RiskAnalystArgument(
                    analysis="c", risk_posture="high"
                ).model_dump_json(),
                "conservative_structured": RiskAnalystArgument(
                    analysis="d", risk_posture="low"
                ).model_dump_json(),
                "neutral_structured": RiskAnalystArgument(
                    analysis="e", risk_posture="moderate"
                ).model_dump_json(),
            },
            "investment_plan_structured": InvestmentPlanJudgment(
                recommendation="Buy",
                narrative="n",
            ).model_dump_json(),
            "trader_investment_plan_structured": TradeProposal(
                decision="HOLD",
                narrative="n",
            ).model_dump_json(),
            "final_trade_decision_structured": RiskAssessment(
                decision="Sell",
                narrative="n",
            ).model_dump_json(),
        }
        out = extract_structured_schedule_literals(final_state)
        self.assertEqual(out["market_outlook"], "bullish")
        self.assertEqual(out["sentiment_outlook"], "neutral")
        self.assertEqual(out["news_outlook"], "mixed")
        self.assertEqual(out["fundamentals_outlook"], "bearish")
        self.assertEqual(out["bull_implied_stance"], "buy")
        self.assertEqual(out["bear_implied_stance"], "sell")
        self.assertEqual(out["risk_aggressive_posture"], "high")
        self.assertEqual(out["risk_conservative_posture"], "low")
        self.assertEqual(out["risk_neutral_posture"], "moderate")
        self.assertEqual(out["investment_recommendation"], "BUY")
        self.assertEqual(out["trader_decision"], "HOLD")
        self.assertEqual(out["risk_decision"], "SELL")

    def test_analyst_outlook_fallback_in_csv(self):
        final_state = {
            "market_report_structured": model_dump_json_with_recovery(
                AnalystReport(report="draft", outlook="mixed"), True
            ),
        }
        out = extract_structured_schedule_literals(final_state)
        self.assertEqual(out["market_outlook"], "fallback")

    def test_bull_bear_implied_stance_fallback_in_csv(self):
        final_state = {
            "investment_debate_state": {
                "bull_structured": model_dump_json_with_recovery(
                    BullBearArgument(analysis="x", implied_stance="neutral"), True
                ),
                "bear_structured": model_dump_json_with_recovery(
                    BullBearArgument(analysis="y", implied_stance="neutral"), True
                ),
            },
        }
        out = extract_structured_schedule_literals(final_state)
        self.assertEqual(out["bull_implied_stance"], "fallback")
        self.assertEqual(out["bear_implied_stance"], "fallback")

    def test_risk_posture_fallback_in_csv(self):
        final_state = {
            "risk_debate_state": {
                "aggressive_structured": model_dump_json_with_recovery(
                    RiskAnalystArgument(analysis="a", risk_posture="moderate"), True
                ),
                "conservative_structured": model_dump_json_with_recovery(
                    RiskAnalystArgument(analysis="b", risk_posture="moderate"), True
                ),
                "neutral_structured": model_dump_json_with_recovery(
                    RiskAnalystArgument(analysis="c", risk_posture="moderate"), True
                ),
            },
        }
        out = extract_structured_schedule_literals(final_state)
        self.assertEqual(out["risk_aggressive_posture"], "fallback")
        self.assertEqual(out["risk_conservative_posture"], "fallback")
        self.assertEqual(out["risk_neutral_posture"], "fallback")
