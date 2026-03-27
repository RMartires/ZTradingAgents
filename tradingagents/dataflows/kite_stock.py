from __future__ import annotations

from datetime import datetime
import json
from typing import List, Dict, Any, Optional

import pandas as pd

from .kite_common import get_kite_session, KiteRateLimitError, maybe_convert_to_kite_rate_limit
from .kite_instruments import get_instrument_mapper


def get_stock_data(
    symbol: str,
    start_date: str,
    end_date: str,
) -> str:
    """
    Retrieve stock price data (OHLCV) for a given ticker symbol.

    Returns the same high-level string/CSV format as `y_finance.get_YFin_data_online`.
    """
    try:
        mapper = get_instrument_mapper()
        resolved = mapper.resolve(symbol)
        kite = get_kite_session().get_client()

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        records: List[Dict[str, Any]] = kite.historical_data(
            resolved["instrument_token"],
            start_dt,
            end_dt,
            interval="day",
        )

        if not records:
            return f"No data found for symbol '{symbol}' between {start_date} and {end_date}"

        df = pd.DataFrame.from_records(records)
        # Kite uses key `date`
        if "date" in df.columns:
            df = df.rename(columns={"date": "Date"})

        # Kite's historical_data returns keys: open, high, low, close, volume
        # Normalize column casing to match yfinance output expectations.
        rename_map = {}
        for k in ["open", "high", "low", "close", "volume"]:
            if k in df.columns:
                rename_map[k] = k.capitalize() if k != "volume" else "Volume"
        if rename_map:
            df = df.rename(columns=rename_map)

        # For compatibility with yfinance, provide Adj Close (we map to Close).
        if "Close" in df.columns and "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"]

        # Keep only expected columns
        keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
        df = df[keep]

        # Use explicit header date range from inputs (not from min/max)
        numeric_columns = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df = df.dropna(subset=["Date"])

        csv_string = df.to_csv(index=False)
        header = f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n"
        header += f"# Total records: {len(df)}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        return header + csv_string

    except Exception as e:
        converted = maybe_convert_to_kite_rate_limit(e)
        if isinstance(converted, KiteRateLimitError):
            raise converted
        raise


def _to_kite_instrument_name(resolved: Dict[str, Any]) -> str:
    """
    Convert our resolved dict to Kite's "EXCHANGE:TRADINGSYMBOL" format.
    Example: "NSE:RELIANCE"
    """
    return f"{resolved['exchange']}:{resolved['tradingsymbol']}"


def get_ltp(symbol: str) -> str:
    """
    Retrieve last traded price (LTP) from Kite.

    Returns:
        Human-readable string for consumption by LLM/tools.
    """
    try:
        mapper = get_instrument_mapper()
        resolved = mapper.resolve(symbol)
        kite = get_kite_session().get_client()

        kite_name = _to_kite_instrument_name(resolved)
        data = kite.ltp(kite_name)
        # Kite returns {"EXCHANGE:SYMBOL": {...}} or similar depending on SDK version.
        payload = data.get(kite_name) if isinstance(data, dict) else None
        if not payload:
            raise ValueError(f"Unexpected Kite ltp response for {kite_name}: {data}")

        last_price = payload.get("last_price")
        return f"## LTP for {symbol.upper()} ({resolved['exchange']}:{resolved['tradingsymbol']}): {last_price}"
    except Exception as e:
        converted = maybe_convert_to_kite_rate_limit(e)
        if isinstance(converted, KiteRateLimitError):
            raise converted
        raise


def get_quote(symbol: str) -> str:
    """
    Retrieve full quote from Kite (depth/ohlc/volume/etc).

    Returns:
        JSON-stringified quote payload.
    """
    try:
        mapper = get_instrument_mapper()
        resolved = mapper.resolve(symbol)
        kite = get_kite_session().get_client()

        kite_name = _to_kite_instrument_name(resolved)
        data = kite.quote(kite_name)
        payload = data.get(kite_name) if isinstance(data, dict) else None
        if payload is None:
            raise ValueError(f"Unexpected Kite quote response for {kite_name}: {data}")

        # Provide JSON for downstream parsers/LLMs.
        return json.dumps(payload, indent=2, default=str)
    except Exception as e:
        converted = maybe_convert_to_kite_rate_limit(e)
        if isinstance(converted, KiteRateLimitError):
            raise converted
        raise

