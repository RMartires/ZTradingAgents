import time
import json

from tradingagents.backtest.structured_literals import model_dump_json_with_recovery
from tradingagents.llm_clients.invoke_fallback import (
    invoke_structured_prompt_or_plain,
)
from tradingagents.schemas import InvestmentPlanJudgment


def create_research_manager(llm, memory, fallback_llm=None):
    def research_manager_node(state) -> dict:
        history = state["investment_debate_state"].get("history", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        investment_debate_state = state["investment_debate_state"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""As the portfolio manager and debate facilitator, your role is to critically evaluate this round of debate and make a definitive decision: align with the bear analyst, the bull analyst, or choose Hold only if it is strongly justified based on the arguments presented.

Summarize the key points from both sides concisely, focusing on the most compelling evidence or reasoning. Your recommendation—Buy, Sell, or Hold—must be clear and actionable. Avoid defaulting to Hold simply because both sides have valid points; commit to a stance grounded in the debate's strongest arguments.

Additionally, develop a detailed investment plan for the trader. This should include:

Your Recommendation: A decisive stance supported by the most convincing arguments.
Rationale: An explanation of why these arguments lead to your conclusion.
Strategic Actions: Concrete steps for implementing the recommendation.
Take into account your past mistakes on similar situations. Use these insights to refine your decision-making and ensure you are learning and improving. Present your analysis conversationally, as if speaking naturally, without special formatting. 

Here are your past reflections on mistakes:
\"{past_memory_str}\"

Here is the debate:
Debate History:
{history}"""
        structured, plain_fallback = invoke_structured_prompt_or_plain(
            llm,
            prompt,
            InvestmentPlanJudgment,
            build_from_text=lambda t: InvestmentPlanJudgment(
                recommendation="Hold",
                rationale="",
                strategic_actions="",
                narrative=(t.strip() or "Structured plan unavailable."),
            ),
            fallback_llm=fallback_llm,
            context="Research Manager",
        )
        content = structured.narrative.strip() or (
            f"Recommendation: {structured.recommendation}. {structured.rationale} {structured.strategic_actions}"
        )

        new_investment_debate_state = {
            "judge_decision": content,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": content,
            "count": investment_debate_state["count"],
        }
        if "bull_structured" in investment_debate_state:
            new_investment_debate_state["bull_structured"] = investment_debate_state[
                "bull_structured"
            ]
        if "bear_structured" in investment_debate_state:
            new_investment_debate_state["bear_structured"] = investment_debate_state[
                "bear_structured"
            ]

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": content,
            "investment_plan_structured": model_dump_json_with_recovery(
                structured, plain_fallback
            ),
        }

    return research_manager_node
