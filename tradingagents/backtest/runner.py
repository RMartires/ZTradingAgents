from __future__ import annotations

import csv
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

OnDayCompleteCallback = Callable[
    [
        str,  # date
        str,  # signal
        Optional[float],  # equity (NAV) at close
        Optional[str],  # error message (empty/None means success)
        Optional[float],  # close (used for NAV)
        Optional[float],  # cash after the day
        Optional[float],  # shares after the day
    ],
    None,
]

from tradingagents.backtest.ledger import PaperLedger
from tradingagents.backtest.prices import fetch_close_for_trade_date
from tradingagents.backtest.signals import resolve_signal
from tradingagents.observability.langfuse_config import (
    get_langfuse_client,
    langfuse_trace_display_name,
    new_langfuse_run_correlation,
    shutdown_langfuse,
)

_log = logging.getLogger(__name__)


def max_drawdown(equities: List[float]) -> float:
    if not equities:
        return 0.0
    peak = equities[0]
    max_dd = 0.0
    for x in equities:
        if x > peak:
            peak = x
        if peak > 0:
            dd = (peak - x) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd


def write_backtest_mvp_artifacts(
    base: Path,
    ticker: str,
    run_id: str,
    initial_cash: float,
    dates_this_run: int,
    equity_rows: List[Dict[str, Any]],
    ledger: PaperLedger,
    *,
    complete: bool,
    last_completed_date: Optional[str] = None,
    langfuse_dates_total: Optional[int] = None,
    write_equity_trades: bool = True,
) -> Dict[str, Any]:
    """Write backtest artifacts for current state.

    When ``write_equity_trades`` is True, writes ``equity.csv`` and ``trades.csv``.
    Always writes ``summary.json``.
    """
    equities = [float(r["equity"]) for r in equity_rows if r.get("close") is not None]
    if not equities and equity_rows:
        equities = [float(r["equity"]) for r in equity_rows]

    initial_eq = float(initial_cash)
    final_eq = equity_rows[-1]["equity"] if equity_rows else initial_eq
    total_return = (final_eq - initial_eq) / initial_eq if initial_eq else 0.0

    executions = sum(
        1
        for t in ledger.trades
        if t.shares_before != t.shares_after or abs(t.cash_before - t.cash_after) > 1e-6
    )

    summary: Dict[str, Any] = {
        "ticker": ticker,
        "run_id": run_id,
        "dates": dates_this_run,
        "dates_completed": len(equity_rows),
        "initial_cash": initial_cash,
        "final_equity": final_eq,
        "total_return": total_return,
        "max_drawdown": max_drawdown(equities) if equities else 0.0,
        "execution_events": executions,
        "output_dir": str(base.resolve()),
        "status": "complete" if complete else "running",
        "last_completed_date": last_completed_date or "",
    }
    if langfuse_dates_total is not None:
        summary["langfuse_dates_total"] = langfuse_dates_total

    if write_equity_trades:
        eq_path = base / "equity.csv"
        with eq_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "date",
                    "signal",
                    "close",
                    "cash",
                    "shares",
                    "equity",
                    "processed_signal",
                ],
            )
            w.writeheader()
            for row in equity_rows:
                w.writerow(
                    {
                        **row,
                        "close": row["close"] if row["close"] is not None else "",
                        "processed_signal": row.get("processed_signal", ""),
                    }
                )

        trades_path = base / "trades.csv"
        with trades_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "trade_date",
                    "signal",
                    "close_price",
                    "shares_before",
                    "shares_after",
                    "cash_before",
                    "cash_after",
                ],
            )
            w.writeheader()
            for t in ledger.trades:
                w.writerow(
                    {
                        "trade_date": t.trade_date,
                        "signal": t.signal,
                        "close_price": t.close_price,
                        "shares_before": t.shares_before,
                        "shares_after": t.shares_after,
                        "cash_before": t.cash_before,
                        "cash_after": t.cash_after,
                    }
                )

    summary_path = base / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_backtest_mvp(
    graph: Any,
    ticker: str,
    dates: List[str],
    *,
    initial_cash: float = 100_000.0,
    buy_fraction: float = 1.0,
    use_llm_signal: bool = False,
    results_dir: Optional[Path] = None,
    portfolio_context: Optional[str] = None,
    use_live_portfolio: bool = False,
    langfuse_meta: Optional[Dict[str, Any]] = None,
    on_day_complete: Optional[OnDayCompleteCallback] = None,
    initial_ledger: Optional[PaperLedger] = None,
    initial_last_close: Optional[float] = None,
    langfuse_dates_total: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run the full agent graph per date, apply paper trades at close, write CSV/JSON.

    Args:
        graph: ``TradingAgentsGraph`` instance (config / data vendors already set).
        ticker: Single symbol.
        dates: Decision dates ``YYYY-MM-DD`` in run order.
        initial_cash: Starting cash.
        buy_fraction: Fraction of cash to deploy on each BUY.
        use_llm_signal: If True, use ``SignalProcessor`` when heuristic is ambiguous.
        results_dir: Output folder; default ``eval_results/<ticker>/backtest_mvp_<id>``.
        portfolio_context: Optional markdown injected into agents; skips Kite when set.
        use_live_portfolio: Passed through to ``propagate`` when ``portfolio_context`` is None.
        langfuse_meta: Optional extras (e.g. llm_provider, quick_think_llm) merged into each
            per-day Langfuse trace input when Langfuse is enabled.
        on_day_complete: Called after each date with ``(date, signal, equity_or_none, error_or_none)``.
        initial_ledger: Optional starting paper ledger for resume.
        initial_last_close: Optional starting close used for NAV when a close is missing.
        langfuse_dates_total: Overrides ``dates_total`` in Langfuse trace metadata (e.g. full schedule size).
    """
    ticker = ticker.strip()
    run_id = uuid.uuid4().hex[:10]
    langfuse_client = get_langfuse_client()
    use_langfuse = langfuse_client is not None
    if results_dir is None:
        base = Path("eval_results") / ticker / f"backtest_mvp_{run_id}"
    else:
        base = Path(results_dir)
    base.mkdir(parents=True, exist_ok=True)

    write_equity_trades = on_day_complete is None

    ledger = initial_ledger if initial_ledger is not None else PaperLedger(cash=float(initial_cash))
    equity_rows: List[Dict[str, Any]] = []
    last_close: Optional[float] = (
        float(initial_last_close) if initial_last_close is not None else None
    )
    trace_dates_total = (
        int(langfuse_dates_total) if langfuse_dates_total is not None else len(dates)
    )

    def _write_snapshot(*, complete: bool, last_completed: Optional[str]) -> Dict[str, Any]:
        return write_backtest_mvp_artifacts(
            base,
            ticker,
            run_id,
            initial_cash,
            len(dates),
            equity_rows,
            ledger,
            complete=complete,
            last_completed_date=last_completed,
            langfuse_dates_total=langfuse_dates_total,
            write_equity_trades=write_equity_trades,
        )

    def _propagate_one_day(d: str, day_index: int) -> tuple[Any, Any]:
        propagate_kw = {"use_live_portfolio": use_live_portfolio}
        if portfolio_context is not None:
            propagate_kw["portfolio_context"] = portfolio_context

        if not use_langfuse:
            return graph.propagate(ticker, d, **propagate_kw)

        from langfuse import propagate_attributes

        corr = new_langfuse_run_correlation(ticker=ticker, trade_date=d)
        trace_display_name = langfuse_trace_display_name(corr.run_suffix)
        trace_input: Dict[str, Any] = {
            "company_name": ticker,
            "trade_date": d,
            "date_index": day_index + 1,
            "dates_total": trace_dates_total,
            "run": "backtest_mvp",
            "backtest_run_id": run_id,
        }
        if langfuse_meta:
            trace_input.update(langfuse_meta)

        trace_kwargs = dict(
            as_type="span",
            name=trace_display_name,
            input=trace_input,
        )
        tc = corr.trace_context
        if tc is not None:
            trace_kwargs["trace_context"] = tc

        trace_cm = langfuse_client.start_as_current_observation(**trace_kwargs)
        root_span = trace_cm.__enter__()
        lp = langfuse_meta or {}
        tags = [
            "backtest_mvp",
            f"ticker:{ticker}",
            f"trade_date:{d}",
            f"backtest_run:{run_id}",
            f"run:{corr.run_suffix}",
            f"llm_provider:{lp.get('llm_provider', '')}",
            f"quick_model:{lp.get('quick_think_llm', '')}",
            f"deep_model:{lp.get('deep_think_llm', '')}",
        ]
        propagate_cm = propagate_attributes(
            trace_name=trace_display_name,
            session_id=corr.session_id,
            user_id=os.getenv("LANGFUSE_USER_ID"),
            tags=tags,
        )
        propagate_cm.__enter__()
        final_state: Any = None
        processed: Any = None
        try:
            final_state, processed = graph.propagate(ticker, d, **propagate_kw)
            return final_state, processed
        finally:
            try:
                exc = sys.exc_info()[1]
                if root_span is not None:
                    if exc is not None:
                        out_payload: Dict[str, Any] = {
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    else:
                        out_payload = {
                            "processed_signal": str(processed)[:200] if processed is not None else "",
                            "final_trade_decision_preview": str(
                                (final_state or {}).get("final_trade_decision") or ""
                            )[:500],
                        }
                    root_span.set_trace_io(input=trace_input, output=out_payload)
                    root_span.update(output=out_payload)
            finally:
                propagate_cm.__exit__(None, None, None)
                trace_cm.__exit__(None, None, None)

    try:
        for day_index, d in enumerate(dates):
            d = str(d).strip()
            try:
                final_state, processed = _propagate_one_day(d, day_index)

                full_text = final_state.get("final_trade_decision") or ""

                signal = resolve_signal(
                    full_text,
                    processed=processed,
                    use_llm=use_llm_signal,
                    signal_processor=graph.signal_processor if use_llm_signal else None,
                )

                close = fetch_close_for_trade_date(ticker, d)
                if close is None:
                    _log.warning("No close price for %s on %s; skipping execution", ticker, d)
                    nav = ledger.equity(last_close) if last_close is not None else ledger.cash
                    equity_rows.append(
                        {
                            "date": d,
                            "signal": signal,
                            "close": None,
                            "cash": ledger.cash,
                            "shares": ledger.shares,
                            "equity": nav,
                            "processed_signal": processed,
                        }
                    )
                    if on_day_complete is not None:
                        on_day_complete(
                            d,
                            signal,
                            float(nav),
                            f"No close price for {ticker} on {d}",
                            None,
                            None,
                            None,
                        )
                    _write_snapshot(complete=False, last_completed=d)
                    continue

                last_close = close
                ledger.apply_signal(signal, close, buy_fraction=buy_fraction, asof_date=d)
                nav = ledger.equity(close)
                equity_rows.append(
                    {
                        "date": d,
                        "signal": signal,
                        "close": close,
                        "cash": ledger.cash,
                        "shares": ledger.shares,
                        "equity": nav,
                        "processed_signal": processed,
                    }
                )
                _log.info(
                    "backtest %s %s signal=%s close=%s nav=%.2f",
                    ticker,
                    d,
                    signal,
                    close,
                    nav,
                )
                if on_day_complete is not None:
                    on_day_complete(
                        d,
                        signal,
                        float(nav),
                        None,
                        float(close),
                        float(ledger.cash),
                        float(ledger.shares),
                    )
                _write_snapshot(complete=False, last_completed=d)
            except Exception as e:
                _log.exception("backtest day failed %s %s", ticker, d)
                err = f"{type(e).__name__}: {e}"
                if on_day_complete is not None:
                    on_day_complete(d, "", None, err, None, None, None)
                _write_snapshot(
                    complete=False,
                    last_completed=equity_rows[-1]["date"] if equity_rows else None,
                )
                continue
    finally:
        if use_langfuse:
            shutdown_langfuse()

    summary = _write_snapshot(
        complete=True,
        last_completed=equity_rows[-1]["date"] if equity_rows else None,
    )

    return {"summary": summary, "equity_rows": equity_rows, "ledger": ledger}
