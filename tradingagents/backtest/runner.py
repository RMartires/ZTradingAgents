from __future__ import annotations

import csv
import json
import logging
import os
import statistics
import sys
import uuid
from datetime import datetime
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
        Optional[Dict[str, str]],  # analysis columns for dates.csv (running metrics)
    ],
    None,
]

from tradingagents.backtest.dates_schedule import (
    SCHEDULE_ANALYSIS_FIELDNAMES,
    empty_schedule_analysis_values,
)
from tradingagents.backtest.structured_literals import extract_structured_schedule_literals
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


def _ledger_fees_for_row(ledger: PaperLedger, traded_today: bool) -> tuple[float, float]:
    """Daily fee on the last ledger trade if ``traded_today``, else 0; plus cumulative fees."""
    cumulative = sum(t.fees_paid for t in ledger.trades)
    if not traded_today or not ledger.trades:
        return 0.0, cumulative
    return float(ledger.trades[-1].fees_paid), cumulative


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


def annualized_return(
    total_return: float,
    equity_rows: List[Dict[str, Any]],
) -> Optional[float]:
    """CAGR-style return over the calendar span covered by ``equity_rows`` (first to last date)."""
    if not equity_rows:
        return None
    d0 = equity_rows[0].get("date")
    d1 = equity_rows[-1].get("date")
    if not d0 or not d1:
        return None
    try:
        t0 = datetime.strptime(str(d0).strip(), "%Y-%m-%d")
        t1 = datetime.strptime(str(d1).strip(), "%Y-%m-%d")
    except ValueError:
        return None
    days = (t1 - t0).days
    if days <= 0:
        return None
    years = days / 365.25
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def sharpe_ratio(
    equity_rows: List[Dict[str, Any]],
    *,
    trading_days_per_year: float = 252.0,
) -> Optional[float]:
    """Daily Sharpe from the equity curve (simple returns, sample stdev)."""
    series: List[float] = []
    for r in equity_rows:
        v = r.get("equity")
        if v is None:
            continue
        try:
            series.append(float(v))
        except (TypeError, ValueError):
            continue
    if len(series) < 3:
        return None
    returns: List[float] = []
    for i in range(1, len(series)):
        prev, cur = series[i - 1], series[i]
        if prev <= 0:
            continue
        returns.append((cur / prev) - 1.0)
    if len(returns) < 2:
        return None
    mean_r = statistics.mean(returns)
    try:
        std_r = statistics.stdev(returns)
    except statistics.StatisticsError:
        return None
    if std_r <= 1e-12:
        return None
    return (mean_r / std_r) * (trading_days_per_year**0.5)


def _equity_series_for_drawdown(equity_rows: List[Dict[str, Any]]) -> List[float]:
    equities = [float(r["equity"]) for r in equity_rows if r.get("close") is not None]
    if not equities and equity_rows:
        equities = [float(r["equity"]) for r in equity_rows]
    return equities


def compute_running_perf_numbers(
    initial_cash: float,
    equity_rows: List[Dict[str, Any]],
    ledger: PaperLedger,
) -> Dict[str, Any]:
    """Performance stats over ``equity_rows`` so far (same math as ``summary.json``)."""
    if not equity_rows:
        return {
            "total_return": 0.0,
            "annualized_return": None,
            "sharpe_ratio": None,
            "max_drawdown": 0.0,
            "total_transaction_costs": float(sum(t.fees_paid for t in ledger.trades)),
            "cost_bps": float(ledger.cost_bps),
        }
    equities = _equity_series_for_drawdown(equity_rows)
    initial_eq = float(initial_cash)
    final_eq = float(equity_rows[-1]["equity"])
    total_return = (final_eq - initial_eq) / initial_eq if initial_eq else 0.0
    total_fees = float(sum(t.fees_paid for t in ledger.trades))
    return {
        "total_return": total_return,
        "annualized_return": annualized_return(total_return, equity_rows),
        "sharpe_ratio": sharpe_ratio(equity_rows),
        "max_drawdown": float(max_drawdown(equities)) if equities else 0.0,
        "total_transaction_costs": total_fees,
        "cost_bps": float(ledger.cost_bps),
    }


def _fmt_schedule_float(x: Optional[float], nd: int = 6) -> str:
    if x is None:
        return ""
    if isinstance(x, (int, float)) and (x != x):  # NaN
        return ""
    return f"{float(x):.{nd}f}"


def build_schedule_analysis_row(
    initial_cash: float,
    equity_rows: List[Dict[str, Any]],
    ledger: PaperLedger,
    *,
    structured_literals: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """String values for ``SCHEDULE_ANALYSIS_FIELDNAMES`` after the latest ``equity_rows`` row."""
    row = empty_schedule_analysis_values()
    if not equity_rows:
        return row
    perf = compute_running_perf_numbers(initial_cash, equity_rows, ledger)
    last = equity_rows[-1]
    fees_day = last.get("fees_day", 0.0)
    cum_fees = last.get("cumulative_fees", perf["total_transaction_costs"])
    ps = last.get("processed_signal", "")
    if ps is not None and not isinstance(ps, str):
        ps = str(ps)
    ps = (ps or "")[:4000]

    row.update(
        {
            "fees_day": _fmt_schedule_float(float(fees_day) if fees_day is not None else 0.0),
            "cumulative_fees": _fmt_schedule_float(
                float(cum_fees) if cum_fees is not None else perf["total_transaction_costs"]
            ),
            "total_return": _fmt_schedule_float(float(perf["total_return"])),
            "annualized_return": _fmt_schedule_float(
                float(perf["annualized_return"]) if perf["annualized_return"] is not None else None
            ),
            "sharpe_ratio": _fmt_schedule_float(
                float(perf["sharpe_ratio"]) if perf["sharpe_ratio"] is not None else None
            ),
            "max_drawdown": _fmt_schedule_float(float(perf["max_drawdown"])),
            "total_transaction_costs": _fmt_schedule_float(float(perf["total_transaction_costs"])),
            "cost_bps": _fmt_schedule_float(float(perf["cost_bps"])),
            "processed_signal": ps,
        }
    )
    if structured_literals:
        for k, v in structured_literals.items():
            if k in row:
                row[k] = (v or "")[:256]
    for k in SCHEDULE_ANALYSIS_FIELDNAMES:
        row.setdefault(k, "")
    return row


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
    initial_eq = float(initial_cash)
    final_eq = equity_rows[-1]["equity"] if equity_rows else initial_eq
    perf = compute_running_perf_numbers(initial_cash, equity_rows, ledger)

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
        "total_return": perf["total_return"],
        "annualized_return": perf["annualized_return"],
        "sharpe_ratio": perf["sharpe_ratio"],
        "max_drawdown": perf["max_drawdown"],
        "cost_bps": perf["cost_bps"],
        "total_transaction_costs": perf["total_transaction_costs"],
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
                    "fees_day",
                    "cumulative_fees",
                    "processed_signal",
                ],
            )
            w.writeheader()
            for row in equity_rows:
                w.writerow(
                    {
                        **row,
                        "close": row["close"] if row["close"] is not None else "",
                        "fees_day": row.get("fees_day", ""),
                        "cumulative_fees": row.get("cumulative_fees", ""),
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
                    "fees_paid",
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
                        "fees_paid": t.fees_paid,
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
    cost_bps: float = 0.0,
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
        cost_bps: Basis points charged on each BUY/SELL notional (0 disables). Ignored if
            ``initial_ledger`` is provided (resume uses the seeded ledger's ``cost_bps``).
        use_llm_signal: If True, use ``SignalProcessor`` when heuristic is ambiguous.
        results_dir: Output folder; default ``eval_results/<ticker>/backtest_mvp_<id>``.
        portfolio_context: Optional markdown injected into agents; skips Kite when set.
        use_live_portfolio: Passed through to ``propagate`` when ``portfolio_context`` is None.
        langfuse_meta: Optional extras (e.g. llm_provider, quick_think_llm) merged into each
            per-day Langfuse trace input when Langfuse is enabled.
        on_day_complete: Called after each date with ``(date, signal, equity, error, close, cash,
            shares, analysis)`` where ``analysis`` is string metrics for ``dates.csv`` (or blanks on error).
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

    ledger = (
        initial_ledger
        if initial_ledger is not None
        else PaperLedger(cash=float(initial_cash), cost_bps=float(cost_bps))
    )
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
                structured_lit = extract_structured_schedule_literals(final_state)

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
                    fd, fc = _ledger_fees_for_row(ledger, traded_today=False)
                    equity_rows.append(
                        {
                            "date": d,
                            "signal": signal,
                            "close": None,
                            "cash": ledger.cash,
                            "shares": ledger.shares,
                            "equity": nav,
                            "fees_day": fd,
                            "cumulative_fees": fc,
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
                            build_schedule_analysis_row(
                                initial_cash,
                                equity_rows,
                                ledger,
                                structured_literals=structured_lit,
                            ),
                        )
                    _write_snapshot(complete=False, last_completed=d)
                    continue

                last_close = close
                ledger.apply_signal(signal, close, buy_fraction=buy_fraction, asof_date=d)
                nav = ledger.equity(close)
                fd, fc = _ledger_fees_for_row(ledger, traded_today=True)
                equity_rows.append(
                    {
                        "date": d,
                        "signal": signal,
                        "close": close,
                        "cash": ledger.cash,
                        "shares": ledger.shares,
                        "equity": nav,
                        "fees_day": fd,
                        "cumulative_fees": fc,
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
                        build_schedule_analysis_row(
                            initial_cash,
                            equity_rows,
                            ledger,
                            structured_literals=structured_lit,
                        ),
                    )
                _write_snapshot(complete=False, last_completed=d)
            except Exception as e:
                _log.exception("backtest day failed %s %s", ticker, d)
                err = f"{type(e).__name__}: {e}"
                if on_day_complete is not None:
                    on_day_complete(
                        d,
                        "",
                        None,
                        err,
                        None,
                        None,
                        None,
                        empty_schedule_analysis_values(),
                    )
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
