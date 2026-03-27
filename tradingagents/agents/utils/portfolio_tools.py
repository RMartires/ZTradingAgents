import json
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from tradingagents.dataflows.kite_common import (
    get_kite_session,
    KiteRateLimitError,
    maybe_convert_to_kite_rate_limit,
)


def _markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    # Very small helper to keep formatting consistent.
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = []
    for r in rows:
        body_lines.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    return "\n".join([header_line, sep_line] + body_lines)


@tool
def get_holdings() -> str:
    """Return current equity holdings (shares and average price) from Kite."""
    try:
        kite = get_kite_session().get_client()
        holdings = kite.holdings() or []
        if not holdings:
            return "## Holdings\n(no holdings found)"

        # Expected fields in Kite holdings objects vary; use best-effort mapping.
        headers = ["Tradingsymbol", "Qty", "AvgPrice", "LastPrice", "P&L"]
        rows: List[List[Any]] = []
        for h in holdings:
            rows.append(
                [
                    h.get("tradingsymbol") or h.get("symbol"),
                    h.get("quantity") or h.get("qty"),
                    h.get("average_price") or h.get("avg_price") or h.get("avgPrice"),
                    h.get("last_price") or h.get("ltp"),
                    h.get("pnl") or h.get("profit_loss") or h.get("unrealised") or h.get("unrealised_pnl"),
                ]
            )

        return "## Holdings\n" + _markdown_table(headers, rows)
    except Exception as e:
        converted = maybe_convert_to_kite_rate_limit(e)
        if isinstance(converted, KiteRateLimitError):
            raise converted
        raise


@tool
def get_positions() -> str:
    """Return current open positions (day + net) from Kite."""
    try:
        kite = get_kite_session().get_client()
        positions = kite.positions() or {}

        day = positions.get("day") if isinstance(positions, dict) else None
        net = positions.get("net") if isinstance(positions, dict) else None

        def format_side(side_name: str, side_rows: Optional[List[Dict[str, Any]]]) -> str:
            if not side_rows:
                return f"### {side_name}\n(no positions)"

            headers = ["Tradingsymbol", "Qty", "AvgPrice", "LastPrice"]
            rows: List[List[Any]] = []
            for p in side_rows:
                rows.append(
                    [
                        p.get("tradingsymbol") or p.get("symbol"),
                        p.get("quantity") or p.get("qty"),
                        p.get("average_price") or p.get("avg_price") or p.get("avgPrice"),
                        p.get("last_price") or p.get("ltp"),
                    ]
                )
            return f"### {side_name}\n" + _markdown_table(headers, rows)

        return "## Positions\n" + format_side("Day", day) + "\n\n" + format_side("Net", net)
    except Exception as e:
        converted = maybe_convert_to_kite_rate_limit(e)
        if isinstance(converted, KiteRateLimitError):
            raise converted
        raise


@tool
def get_available_funds() -> str:
    """Return available funds and margin summary from Kite."""
    try:
        kite = get_kite_session().get_client()
        margins = kite.margins() or {}

        def fmt_segment(seg_name: str) -> str:
            seg = margins.get(seg_name, {}) if isinstance(margins, dict) else {}
            enabled = seg.get("enabled", None)
            net = seg.get("net", None)

            available = seg.get("available", {}) or {}
            utilised = seg.get("utilised", {}) or {}

            cash = available.get("cash", None)
            live_balance = available.get("live_balance", None)
            intraday_payin = available.get("intraday_payin", None)

            debits = utilised.get("debits", None)
            span = utilised.get("span", None)
            exposure = utilised.get("exposure", None)

            rows = [
                ["enabled", enabled],
                ["net", net],
                ["available.cash", cash],
                ["available.live_balance", live_balance],
                ["available.intraday_payin", intraday_payin],
                ["utilised.debits", debits],
                ["utilised.span", span],
                ["utilised.exposure", exposure],
            ]

            headers = ["Field", "Value"]
            return f"### {seg_name.title()}\n" + _markdown_table(headers, rows)

        if not margins:
            return "## Funds and margins\n(no margin data)"

        parts = []
        for seg in ["equity", "commodity"]:
            if isinstance(margins, dict) and seg in margins:
                parts.append(fmt_segment(seg))

        # Include raw payload for debugging/LLM if desired
        raw = json.dumps(margins, indent=2, default=str)
        return "## Funds and margins\n" + "\n\n".join(parts) + "\n\n### Raw (debug)\n```json\n" + raw + "\n```"
    except Exception as e:
        converted = maybe_convert_to_kite_rate_limit(e)
        if isinstance(converted, KiteRateLimitError):
            raise converted
        raise

