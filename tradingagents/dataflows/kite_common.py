import os
import time
from dataclasses import dataclass
from typing import Optional, Any

# Kite Connect `historical_data` (daily interval): Zerodha caps the requested range at
# this many calendar days between from_date and to_date.
KITE_HISTORICAL_MAX_INTERVAL_DAYS = 2000


class KiteAuthError(RuntimeError):
    """Raised when Kite API credentials are missing or invalid."""


class KiteRateLimitError(RuntimeError):
    """Raised when Kite enforces rate limits and we should fall back."""


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Best-effort detection of rate-limit style failures."""
    msg = str(exc).lower()
    return any(
        s in msg
        for s in (
            "rate limit",
            "rate-limit",
            "too many requests",
            "429",
            "exceeded",
        )
    )


@dataclass
class KiteSession:
    """
    Thin auth wrapper around Kite Connect.

    Notes:
    - Kite Connect access_token typically expires daily at ~6 AM.
    - For this integration, we assume the user provides the current token via env vars.
    """

    api_key: str
    access_token: str
    _kite: Optional[Any] = None  # lazy-init KiteConnect client

    @classmethod
    def from_env(cls) -> "KiteSession":
        api_key = os.getenv("KITE_API_KEY", "").strip()
        access_token = os.getenv("KITE_ACCESS_TOKEN", "").strip()
        if not api_key or not access_token:
            raise KiteAuthError(
                "Missing Kite credentials. Please set KITE_API_KEY and KITE_ACCESS_TOKEN."
            )
        return cls(api_key=api_key, access_token=access_token)

    def get_client(self):
        """Return a lazily initialized KiteConnect client."""
        if self._kite is not None:
            return self._kite

        try:
            # Import lazily so importing this module doesn't hard-require kiteconnect.
            from kiteconnect import KiteConnect

            kite = KiteConnect(api_key=self.api_key)
            kite.set_access_token(self.access_token)
            self._kite = kite
            return kite
        except Exception as e:  # pragma: no cover (depends on kiteconnect runtime)
            if _is_rate_limit_error(e):
                raise KiteRateLimitError(str(e)) from e
            raise KiteAuthError(str(e)) from e


_session: Optional[KiteSession] = None


def get_kite_session(refresh: bool = False) -> KiteSession:
    """
    Get a module-level KiteSession.

    The token is assumed to come from env vars, so `refresh=True` re-reads env vars.
    """
    global _session
    if refresh or _session is None:
        _session = KiteSession.from_env()
    return _session


def is_kite_configured() -> bool:
    """True if required Kite env vars are set."""
    return bool(os.getenv("KITE_API_KEY", "").strip() and os.getenv("KITE_ACCESS_TOKEN", "").strip())


def maybe_convert_to_kite_rate_limit(exc: BaseException) -> BaseException:
    """Convert a rate-limit-looking exception to KiteRateLimitError (otherwise return original)."""
    if _is_rate_limit_error(exc):
        return KiteRateLimitError(str(exc))
    return exc

