import os
import time
from typing import Dict, Any, Optional, Tuple

import pandas as pd

from .kite_common import get_kite_session, KiteRateLimitError
from .config import get_config


def _normalize_symbol(symbol: str) -> Tuple[str, Optional[str]]:
    """
    Normalize common user inputs:
    - "RELIANCE.NS" -> ("RELIANCE", "NSE")
    - "RELIANCE.BO" -> ("RELIANCE", "BSE")
    - "RELIANCE" -> ("RELIANCE", None)
    """
    raw = symbol.strip().upper()
    if "." not in raw:
        return raw, None

    base, suffix = raw.split(".", 1)
    suffix = suffix.strip().upper()

    if suffix in ("NS", "NSE"):
        return base, "NSE"
    if suffix in ("BO", "BSE"):
        return base, "BSE"

    # Unknown suffix: keep base and let caller decide exchange.
    return base, None


class KiteInstrumentMapper:
    """
    Resolve a trading symbol into Kite's numeric `instrument_token`.

    Kite requires `instrument_token` for historical data and WebSocket.
    We also cache Kite's instrument master list locally to avoid repeated downloads.
    """

    def __init__(self, cache_ttl_seconds: int = 24 * 60 * 60):
        cfg = get_config()
        self._cache_dir = cfg.get("data_cache_dir", os.path.join(cfg["project_dir"], "dataflows/data_cache"))
        os.makedirs(self._cache_dir, exist_ok=True)
        self._cache_ttl_seconds = cache_ttl_seconds

        # In-memory cache: exchange -> DataFrame
        self._instruments_cache: Dict[str, pd.DataFrame] = {}

    def _cache_path(self, exchange: str) -> str:
        return os.path.join(self._cache_dir, f"kite_instruments_{exchange}.csv")

    def _cache_age_seconds(self, path: str) -> Optional[float]:
        if not os.path.exists(path):
            return None
        return time.time() - os.path.getmtime(path)

    def _load_instruments(self, exchange: str) -> pd.DataFrame:
        if exchange in self._instruments_cache:
            return self._instruments_cache[exchange]

        path = self._cache_path(exchange)
        age = self._cache_age_seconds(path)
        cache_fresh = age is not None and age < self._cache_ttl_seconds

        if cache_fresh:
            df = pd.read_csv(path)
            self._instruments_cache[exchange] = df
            return df

        # Refresh from Kite.
        kite = get_kite_session().get_client()
        # Kite uses exchange identifiers like "NSE", "BSE".
        instruments = kite.instruments(exchange)

        df = pd.DataFrame(instruments)

        # Persist for later calls.
        try:
            df.to_csv(path, index=False)
        except Exception:
            # Cache is best-effort; if writing fails, still return df.
            pass

        self._instruments_cache[exchange] = df
        return df

    def resolve(self, symbol: str, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Resolve a symbol into the minimal dict needed by our kite data functions.

        Returns:
            {"instrument_token": int, "tradingsymbol": str, "exchange": str}
        """
        tradingsymbol, hinted_exchange = _normalize_symbol(symbol)
        effective_exchange = hinted_exchange or exchange

        df = self._load_instruments(effective_exchange)
        if "tradingsymbol" not in df.columns:
            raise ValueError(f"Kite instruments cache missing 'tradingsymbol' for exchange={effective_exchange}")

        # Prefer equities (EQ) if the dataset includes it.
        mask_base = df["tradingsymbol"].astype(str).str.upper() == tradingsymbol.upper()
        if "instrument_type" in df.columns:
            mask_eq = mask_base & (df["instrument_type"].astype(str) == "EQ")
            matches = df.loc[mask_eq]
            if not matches.empty:
                row = matches.iloc[0]
            else:
                matches = df.loc[mask_base]
                if matches.empty:
                    raise ValueError(f"No Kite instrument match for symbol={symbol} exchange={effective_exchange}")
                row = matches.iloc[0]
        else:
            matches = df.loc[mask_base]
            if matches.empty:
                raise ValueError(f"No Kite instrument match for symbol={symbol} exchange={effective_exchange}")
            row = matches.iloc[0]

        if "instrument_token" not in row:
            raise ValueError(f"Kite instrument row missing instrument_token for symbol={symbol} exchange={effective_exchange}")

        return {
            "instrument_token": int(row["instrument_token"]),
            "tradingsymbol": str(row.get("tradingsymbol", tradingsymbol)),
            "exchange": effective_exchange,
        }


_mapper: Optional[KiteInstrumentMapper] = None


def get_instrument_mapper() -> KiteInstrumentMapper:
    global _mapper
    if _mapper is None:
        _mapper = KiteInstrumentMapper()
    return _mapper

