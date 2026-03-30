from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, timedelta
from typing import Optional

from tradingagents.dataflows.interface import route_to_vendor

_log = logging.getLogger(__name__)


def parse_close_from_vendor_block(block: str, prefer_date: str) -> Optional[float]:
    """
    Parse the string returned by ``get_stock_data`` vendors (header lines + CSV).

    Picks the row whose Date equals ``prefer_date``; if missing, uses the last row
    (e.g. previous session in range).
    """
    if not block or "No data found" in block:
        return None

    # Strip comment lines; remainder should be CSV with header
    lines = []
    for line in block.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)

    if not lines:
        return None

    text = "\n".join(lines)
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        return None

    fn = list(reader.fieldnames)
    fieldnames_lower = {f.lower(): f for f in fn if f}
    close_key = fieldnames_lower.get("close")
    if close_key is None:
        return None

    date_key = fieldnames_lower.get("date")
    if date_key is None and fn:
        # yfinance ``DataFrame.to_csv()`` uses the index as an unnamed first column
        date_key = fn[0]
    # ``date_key`` may be ``""`` (empty header); that is valid for DictReader
    if date_key is None:
        return None

    rows = list(reader)
    if not rows:
        return None

    prefer = prefer_date.strip()[:10]
    for row in rows:
        raw_d = str(row.get(date_key) or "").strip()
        d = raw_d[:10]
        if d == prefer:
            try:
                return float(row[close_key])
            except (TypeError, ValueError):
                return None

    # Fallback: last row (latest date in range)
    last = rows[-1]
    try:
        return float(last[close_key])
    except (TypeError, ValueError):
        return None


def fetch_close_for_trade_date(symbol: str, trade_date: str) -> Optional[float]:
    """
    Fetch daily close for ``trade_date`` using configured stock data vendor.

    Uses ``start=trade_date`` and ``end=trade_date + 1 day`` so yfinance's
    exclusive end boundary still includes the session.
    """
    try:
        d = datetime.strptime(trade_date.strip(), "%Y-%m-%d")
    except ValueError:
        _log.warning("Invalid trade_date %r", trade_date)
        return None

    end_exc = (d + timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        block = route_to_vendor("get_stock_data", symbol, trade_date, end_exc)
    except Exception as e:
        _log.warning("get_stock_data failed for %s %s: %s", symbol, trade_date, e)
        return None

    return parse_close_from_vendor_block(block, trade_date)
