import functools
import time
import json

from langchain_core.messages import AIMessage

from tradingagents.backtest.structured_literals import model_dump_json_with_recovery
from tradingagents.llm_clients.invoke_fallback import (
    invoke_structured_messages_or_plain,
)
from tradingagents.schemas import TradeProposal, append_final_tx_line_if_missing


def create_trader(llm, memory, fallback_llm=None):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        portfolio_context = state.get("portfolio_context", "").strip()
        portfolio_section = (
            f"\n\nCurrent Portfolio Context (holdings/positions/funds):\n{portfolio_context}"
            if portfolio_context
            else ""
        )

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        if past_memories:
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"
        else:
            past_memory_str = "No past memories found."

        context = {
            "role": "user",
            "content": f"Based on a comprehensive analysis by a team of analysts, here is an investment plan tailored for {company_name}. This plan incorporates insights from current technical market trends, macroeconomic indicators, and social media sentiment. Use this plan as a foundation for evaluating your next trading decision.\n\nProposed Investment Plan: {investment_plan}{portfolio_section}\n\nLeverage these insights to make an informed and strategic decision.",
        }

        messages = [
            {
                "role": "system",
                "content": f"""You are a trading agent analyzing market data to make investment decisions. Based on your analysis, provide a specific recommendation to buy, sell, or hold. End with a firm decision and always conclude your response with 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**' to confirm your recommendation. Do not forget to utilize lessons from past decisions to learn from your mistakes. Here is some reflections from similar situatiosn you traded in and the lessons learned: {past_memory_str}{portfolio_section}""",
            },
            context,
        ]

        structured, plain_fallback = invoke_structured_messages_or_plain(
            llm,
            messages,
            TradeProposal,
            build_from_text=lambda t: TradeProposal(
                decision="HOLD",
                rationale="",
                narrative=(t.strip() or "FINAL TRANSACTION PROPOSAL: **HOLD**"),
            ),
            fallback_llm=fallback_llm,
            context="Trader",
        )
        narrative = append_final_tx_line_if_missing(
            structured.narrative, structured.decision
        )
        return {
            "messages": [AIMessage(content=narrative)],
            "trader_investment_plan": narrative,
            "trader_investment_plan_structured": model_dump_json_with_recovery(
                structured, plain_fallback
            ),
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
