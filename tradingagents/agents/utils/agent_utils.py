from langchain_core.messages import HumanMessage, RemoveMessage
from typing import Any, Iterable, List, Sequence

from tradingagents.dataflows.config import get_config

# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import (
    get_stock_data
)
from tradingagents.agents.utils.technical_indicators_tools import (
    get_indicators
)
from tradingagents.agents.utils.fundamental_data_tools import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_insider_transactions,
    get_global_news
)

def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages


def _estimate_tokens(text: str) -> int:
    # Heuristic: ~4 characters per token for English-ish text.
    # (Good enough for preventing accidental context-window blowups from huge tool outputs.)
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _message_to_text(m: Any) -> str:
    # Supports LangChain BaseMessage (has `.content`) and simple (role, content) tuples.
    try:
        content = getattr(m, "content")
    except Exception:
        content = None

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Multimodal / structured content: best-effort stringify.
        return "\n".join(str(x) for x in content)

    if isinstance(m, tuple) and len(m) == 2 and isinstance(m[1], str):
        return m[1]

    return str(m)


def limit_messages_for_llm_context(messages: Sequence[Any]) -> list[Any]:
    """Trim message history to fit an approximate input-token budget.

    Budget is sourced from config key ``llm_max_input_tokens`` (settable via env in `main.py`).
    We keep the most recent messages and drop/truncate older content first.
    """
    try:
        cfg = get_config()
    except Exception:
        cfg = {}

    max_in = cfg.get("llm_max_input_tokens")
    try:
        max_in_int = int(max_in) if max_in is not None else 0
    except Exception:
        max_in_int = 0

    if max_in_int <= 0:
        return list(messages)

    # Walk backwards, keeping newest messages until we hit budget.
    kept_rev: list[Any] = []
    used = 0

    for m in reversed(list(messages)):
        text = _message_to_text(m)
        t = _estimate_tokens(text)

        if used + t <= max_in_int:
            kept_rev.append(m)
            used += t
            continue

        # If nothing has been kept yet, keep a truncated version of the newest message.
        if not kept_rev:
            remaining = max(1, max_in_int - used)
            # convert remaining token budget to char budget
            max_chars = remaining * 4
            truncated = text[:max_chars]
            try:
                # LangChain messages are usually pydantic-ish and allow assignment.
                if hasattr(m, "content"):
                    m.content = truncated  # type: ignore[attr-defined]
                    kept_rev.append(m)
                else:
                    kept_rev.append(m)
            except Exception:
                kept_rev.append(m)
            break

        # Otherwise stop; we've filled the budget with newer messages.
        break

    return list(reversed(kept_rev))


        