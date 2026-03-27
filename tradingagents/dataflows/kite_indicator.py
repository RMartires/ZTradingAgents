from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from dateutil.relativedelta import relativedelta
import pandas as pd
from stockstats import wrap

from .kite_common import (
    KITE_HISTORICAL_MAX_INTERVAL_DAYS,
    get_kite_session,
    maybe_convert_to_kite_rate_limit,
)
from .kite_instruments import get_instrument_mapper
from .config import get_config
from .stockstats_utils import _clean_dataframe


BEST_IND_PARAMS: Dict[str, str] = {
    # Moving Averages
    "close_50_sma": (
        "50 SMA: A medium-term trend indicator. "
        "Usage: Identify trend direction and serve as dynamic support/resistance. "
        "Tips: It lags price; combine with faster indicators for timely signals."
    ),
    "close_200_sma": (
        "200 SMA: A long-term trend benchmark. "
        "Usage: Confirm overall market trend and identify golden/death cross setups. "
        "Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries."
    ),
    "close_10_ema": (
        "10 EMA: A responsive short-term average. "
        "Usage: Capture quick shifts in momentum and potential entry points. "
        "Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals."
    ),
    # MACD Related
    "macd": (
        "MACD: Computes momentum via differences of EMAs. "
        "Usage: Look for crossovers and divergence as signals of trend changes. "
        "Tips: Confirm with other indicators in low-volatility or sideways markets."
    ),
    "macds": (
        "MACD Signal: An EMA smoothing of the MACD line. "
        "Usage: Use crossovers with the MACD line to trigger trades. "
        "Tips: Should be part of a broader strategy to avoid false positives."
    ),
    "macdh": (
        "MACD Histogram: Shows the gap between the MACD line and its signal. "
        "Usage: Visualize momentum strength and spot divergence early. "
        "Tips: Can be volatile; complement with additional filters in fast-moving markets."
    ),
    # Momentum Indicators
    "rsi": (
        "RSI: Measures momentum to flag overbought/oversold conditions. "
        "Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. "
        "Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis."
    ),
    # Volatility Indicators
    "boll": (
        "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. "
        "Usage: Acts as a dynamic benchmark for price movement. "
        "Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals."
    ),
    "boll_ub": (
        "Bollinger Upper Band: Typically 2 standard deviations above the middle line. "
        "Usage: Signals potential overbought conditions and breakout zones. "
        "Tips: Confirm signals with other tools; prices may ride the band in strong trends."
    ),
    "boll_lb": (
        "Bollinger Lower Band: Typically 2 standard deviations below the middle line. "
        "Usage: Indicates potential oversold conditions. "
        "Tips: Use additional analysis to avoid false reversal signals."
    ),
    "atr": (
        "ATR: Averages true range to measure volatility. "
        "Usage: Set stop-loss levels and adjust position sizes based on current market volatility. "
        "Tips: It's a reactive measure, so use it as part of a broader risk management strategy."
    ),
    # Volume-Based Indicators
    "vwma": (
        "VWMA: A moving average weighted by volume. "
        "Usage: Confirm trends by integrating price action with volume data. "
        "Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses."
    ),
    "mfi": (
        "MFI: The Money Flow Index is a momentum indicator that uses both price and volume to measure buying and selling pressure. "
        "Usage: Identify overbought (>80) or oversold (<20) conditions and confirm the strength of trends or reversals. "
        "Tips: Use alongside RSI or MACD to confirm signals; divergence between price and MFI can indicate potential reversals."
    ),
}


def _get_kite_ohlcv(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch OHLCV data from Kite and return a DataFrame with:
    Date, Open, High, Low, Close, Volume
    """
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
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

    df = pd.DataFrame.from_records(records)
    if "date" in df.columns:
        df = df.rename(columns={"date": "Date"})

    # Normalize column casing (kite returns lower-case names)
    rename_map = {}
    for k in ["open", "high", "low", "close", "volume"]:
        if k in df.columns:
            rename_map[k] = k.capitalize() if k != "volume" else "Volume"
    if rename_map:
        df = df.rename(columns=rename_map)

    if "Close" in df.columns and "Adj Close" not in df.columns:
        # stockstats doesn't require Adj Close, but keep compatibility if present
        df["Adj Close"] = df["Close"]

    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep]
    return df


def _get_stockstats_indicator_bulk(symbol: str, indicator: str, curr_date: str) -> Dict[str, str]:
    """
    Compute the indicator for all trading days in a cached OHLCV range.

    Returns a dict mapping YYYY-MM-DD -> indicator_value (as string).
    """
    cfg = get_config()
    cache_dir = cfg.get("data_cache_dir", "dataflows/data_cache")

    end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    ideal_start = end_dt - relativedelta(years=15)
    max_span_start = end_dt - timedelta(
        days=KITE_HISTORICAL_MAX_INTERVAL_DAYS - 1
    )
    start_dt = max(ideal_start, max_span_start)

    start_date_str = start_dt.strftime("%Y-%m-%d")
    end_date_str = end_dt.strftime("%Y-%m-%d")

    # Local import to keep this module lightweight if unused.
    import os

    os.makedirs(cache_dir, exist_ok=True)
    data_file = os.path.join(cache_dir, f"{symbol}-Kite-data-{start_date_str}-{end_date_str}.csv")

    if os.path.exists(data_file):
        data = pd.read_csv(data_file, on_bad_lines="skip")
    else:
        data = _get_kite_ohlcv(symbol, start_date_str, end_date_str)
        if data.empty:
            return {}
        data.to_csv(data_file, index=False)

    data = _clean_dataframe(data)
    df = wrap(data)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    # Trigger stockstats indicator calc
    _ = df[indicator]

    result_dict: Dict[str, str] = {}
    for _, row in df.iterrows():
        date_str = str(row["Date"])
        v = row.get(indicator)
        if pd.isna(v):
            result_dict[date_str] = "N/A"
        else:
            result_dict[date_str] = str(v)

    return result_dict


def get_indicators(
    symbol: str,
    indicator: str,
    curr_date: str,
    look_back_days: int = 30,
) -> str:
    """
    Retrieve a single technical indicator window around `curr_date`.
    Uses Kite OHLCV + stockstats to compute indicators.
    """
    indicator = indicator.strip().lower()
    if indicator not in BEST_IND_PARAMS:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: {list(BEST_IND_PARAMS.keys())}"
        )

    try:
        indicator_data = _get_stockstats_indicator_bulk(symbol, indicator, curr_date)
        if not indicator_data:
            return f"## {indicator} values from N/A to {curr_date}:\n\nN/A: Not a trading day (weekend or holiday)\n\n{BEST_IND_PARAMS.get(indicator, 'No description available.')}"

        curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        before = curr_date_dt - relativedelta(days=look_back_days)

        # Build the indicator string for each day in the window (including non-trading days)
        current_dt = curr_date_dt
        ind_string = ""
        while current_dt >= before:
            date_str = current_dt.strftime("%Y-%m-%d")
            if date_str in indicator_data:
                indicator_value = indicator_data[date_str]
            else:
                indicator_value = "N/A: Not a trading day (weekend or holiday)"

            ind_string += f"{date_str}: {indicator_value}\n"
            current_dt = current_dt - relativedelta(days=1)

        result_str = (
            f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {curr_date}:\n\n"
            + ind_string
            + "\n\n"
            + BEST_IND_PARAMS.get(indicator, "No description available.")
        )
        return result_str
    except Exception as e:
        converted = maybe_convert_to_kite_rate_limit(e)
        # route_to_vendor will only fall back on KiteRateLimitError
        from .kite_common import KiteRateLimitError as _KiteRateLimitError

        if isinstance(converted, _KiteRateLimitError):
            raise converted
        raise

