from __future__ import annotations

import re
from typing import Any, Optional


def normalize_signal_heuristic(text: str) -> Optional[str]:
    """
    Extract BUY / SELL / HOLD from analyst-style prose without an LLM.

    Prefers lines mentioning FINAL TRANSACTION PROPOSAL, then the last
    standalone BUY/SELL/HOLD token.
    """
    if not text or not str(text).strip():
        return None
    upper = str(text).upper()

    m = re.search(
        r"FINAL\s+TRANSACTION\s+PROPOSAL\s*:?\s*\*?\*?\s*(BUY|SELL|HOLD)\b",
        upper,
    )
    if m:
        return m.group(1)

    matches = list(re.finditer(r"\b(BUY|SELL|HOLD)\b", upper))
    if matches:
        return matches[-1].group(1)

    return None


def _canonical_from_processed(processed: str) -> Optional[str]:
    if not processed:
        return None
    p = processed.strip().upper()
    for token in ("BUY", "SELL", "HOLD"):
        if p == token or p.startswith(token + " ") or p.startswith(token + "\n"):
            return token
        if token in p and len(p) <= 20:
            return token
    return None


def resolve_signal(
    full_text: str,
    *,
    processed: Optional[str] = None,
    use_llm: bool = False,
    signal_processor: Optional[Any] = None,
) -> str:
    """
    Resolve a canonical BUY/SELL/HOLD for execution and schedule ``final_signal``.

    Order (intentional):

    1. **``processed``** — output of ``TradingAgentsGraph.process_signal`` (quick LLM
       extraction from ``final_trade_decision``). Prefer this first so the traded signal
       matches the ``processed_signal`` column and avoids spurious BUY/SELL tokens in debate
       text winning over the dedicated extractor.

    2. **Heuristic** on ``full_text`` — regex / last-token fallback when ``processed`` is
       empty or not canonical.

    3. Optional **second LLM** pass via ``SignalProcessor.process_signal`` when
       ``use_llm`` is True (backtest ``--use-llm-signal``).
    """
    if processed:
        c = _canonical_from_processed(processed)
        if c in ("BUY", "SELL", "HOLD"):
            return c

    h = normalize_signal_heuristic(full_text)
    if h in ("BUY", "SELL", "HOLD"):
        return h

    if use_llm and signal_processor is not None:
        try:
            out = str(signal_processor.process_signal(full_text)).strip().upper()
        except Exception:
            out = ""
        for token in ("BUY", "SELL", "HOLD"):
            if token in out:
                return token

    return "HOLD"
