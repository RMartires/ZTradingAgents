#!/usr/bin/env python3
"""
MVP historical backtest: run TradingAgentsGraph per date (no live portfolio),
execute paper trades at close, write eval_results/<ticker>/backtest_mvp_<id>/.

For one-off runs (no ``--dates-csv``), `equity.csv`, `trades.csv`, and `summary.json` are refreshed.
When ``--dates-csv`` is provided, the schedule CSV becomes the source of truth for progress and portfolio state:
columns `date`, `processed`, `final_signal`, `close`, `cash`, `shares`, `equity`, `error` (atomic rewrites).
Saturday/Sunday rows are not backtested: they are marked ``processed=true`` with ``error=not trading day``.
In this mode, the per-run output folder only updates `summary.json`.

Usage (from repo root):
  .venv/bin/python scripts/backtest_mvp.py --ticker RELIANCE --dates 2024-05-03,2024-05-10
  .venv/bin/python scripts/backtest_mvp.py --ticker RELIANCE.NS --dates-file dates.txt
  .venv/bin/python scripts/backtest_mvp.py --ticker X --dates-csv schedule.csv --dates 2024-05-03,2024-05-04
  # (creates schedule.csv if missing from --dates / --dates-file; later runs skip rows with processed=true)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cli.stats_handler import StatsCallbackHandler
from tradingagents.backtest.dates_schedule import (
    empty_schedule_analysis_values,
    is_row_processed,
    last_successful_ledger_state,
    pending_schedule_dates,
    read_dates_schedule,
    update_schedule_row,
    write_dates_schedule_atomic,
)
from tradingagents.observability.langfuse_config import get_langfuse_client, get_langfuse_handler


def _load_dotenv(path: Path | None = None) -> None:
    env_path = path or (ROOT / ".env")
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            os.environ.setdefault(key, value)


def _build_config() -> dict:
    from tradingagents.default_config import DEFAULT_CONFIG

    cfg = DEFAULT_CONFIG.copy()
    # Match main.py defaults so OpenRouter + OPENROUTER_API_KEY work without LLM_PROVIDER in env.
    cfg["llm_provider"] = os.getenv("LLM_PROVIDER", "openrouter")
    cfg["quick_think_llm"] = os.getenv("QUICK_THINK_LLM", "openrouter/free")
    cfg["deep_think_llm"] = os.getenv("DEEP_THINK_LLM", "openrouter/free")
    if os.getenv("BACKEND_URL", "").strip():
        cfg["backend_url"] = os.getenv("BACKEND_URL", "").strip()
    elif os.getenv("OPENROUTER_BASE_URL", "").strip():
        cfg["backend_url"] = os.getenv("OPENROUTER_BASE_URL", "").strip()
    _md = os.getenv("MAX_DEBATE_ROUNDS", "").strip()
    if _md:
        try:
            cfg["max_debate_rounds"] = int(_md)
        except ValueError:
            pass
    _mr = os.getenv("MAX_RISK_DISCUSS_ROUNDS", "").strip()
    if _mr:
        try:
            cfg["max_risk_discuss_rounds"] = int(_mr)
        except ValueError:
            pass
    if os.getenv("LLM_MAX_RETRIES", "").strip():
        cfg["llm_max_retries"] = int(os.getenv("LLM_MAX_RETRIES", "2"))
    if os.getenv("LLM_TIMEOUT", "").strip():
        cfg["llm_timeout"] = float(os.getenv("LLM_TIMEOUT", "600"))
    if os.getenv("LLM_RATE_LIMIT_RPM", "").strip():
        cfg["llm_rate_limit_rpm"] = float(os.getenv("LLM_RATE_LIMIT_RPM", "0"))
    if os.getenv("LLM_MAX_TOKENS", "").strip():
        cfg["llm_max_tokens"] = int(os.getenv("LLM_MAX_TOKENS", "8192"))
    # Optional: LLM_STRUCTURED_TEMPERATURE — schemas.outputs structured invokes only (invoke_fallback).

    # Paper backtest: basis points per BUY/SELL notional. Re-read from env here so ``.env``
    # (loaded in ``main()`` before this runs) wins over ``DEFAULT_CONFIG`` frozen at import.
    _bc = os.getenv("BACKTEST_COST_BPS", "").strip()
    if _bc != "":
        try:
            cfg["backtest_cost_bps"] = float(_bc)
        except ValueError:
            pass

    # Configure data vendors (default uses yfinance, no extra API keys needed)
    cfg["data_vendors"] = {
        "core_stock_apis": "kite",
        "technical_indicators": "kite",
        "fundamental_data": "yfinance",
        "news_data": "alpha_vantage",
    }

    return cfg


def _parse_dates(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    if args.dates:
        out.extend(d.strip() for d in args.dates.split(",") if d.strip())
    if args.dates_file:
        p = Path(args.dates_file)
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    seen = set()
    unique = []
    for d in out:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique


_WEEKEND_SKIP_ERROR = "not trading day"


def _is_weekend_ymd(date_str: str) -> bool:
    """True if ``YYYY-MM-DD`` is Saturday or Sunday (calendar). Invalid dates return False."""
    try:
        d = datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
    except ValueError:
        return False
    return d.weekday() >= 5


def _mark_weekend_rows_skipped(
    schedule_path: Path,
    schedule_rows: list[dict[str, str]],
) -> None:
    """Mark unprocessed weekend rows as processed with ``not trading day``; persist if any changed."""
    changed = False
    for r in schedule_rows:
        d = str(r.get("date", "")).strip()
        if not d or is_row_processed(r.get("processed")):
            continue
        if not _is_weekend_ymd(d):
            continue
        update_schedule_row(
            schedule_rows,
            d,
            processed=True,
            final_signal="",
            equity="",
            error=_WEEKEND_SKIP_ERROR,
            close="",
            cash="",
            shares="",
            analysis=empty_schedule_analysis_values(),
        )
        changed = True
    if changed:
        write_dates_schedule_atomic(schedule_path, schedule_rows)
        logging.info(
            "Marked weekend row(s) as skipped (%r) in %s",
            _WEEKEND_SKIP_ERROR,
            schedule_path,
        )


def main() -> int:
    _load_dotenv()

    parser = argparse.ArgumentParser(description="TradingAgents historical backtest MVP")
    parser.add_argument("--ticker", required=True, help="Single symbol (e.g. RELIANCE or RELIANCE.NS)")
    parser.add_argument("--dates", default="", help="Comma-separated YYYY-MM-DD")
    parser.add_argument("--dates-file", default="", help="File with one date per line")
    parser.add_argument(
        "--dates-csv",
        default="",
        help="CSV schedule (date,processed,final_signal,equity,error). "
        "Runs pending rows only; updated after each date. If missing, created from --dates / --dates-file.",
    )
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--buy-fraction", type=float, default=1.0)
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=None,
        help="Transaction cost in bps of notional per BUY/SELL (overrides BACKTEST_COST_BPS / config)",
    )
    parser.add_argument("--use-llm-signal", action="store_true", help="Use SignalProcessor when heuristic fails")
    parser.add_argument("--debug", action="store_true", help="LangGraph debug stream (verbose)")
    parser.add_argument("--results-dir", default="", help="Override output directory base")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    bootstrap_dates = _parse_dates(args)
    schedule_rows: list[dict[str, str]] | None = None
    schedule_path: Path | None = None
    langfuse_dates_total: int | None = None
    seed_ledger = None
    seed_last_close: float | None = None
    effective_initial_cash = args.initial_cash

    if args.dates_csv.strip():
        schedule_path = Path(args.dates_csv.strip())
        if not schedule_path.is_file():
            if not bootstrap_dates:
                logging.error(
                    "--dates-csv %s not found; provide --dates and/or --dates-file to create it",
                    schedule_path,
                )
                return 2
            initial_rows = [
                {
                    "date": d,
                    "processed": "",
                    "final_signal": "",
                    "equity": "",
                    "error": "",
                    **empty_schedule_analysis_values(),
                }
                for d in bootstrap_dates
            ]
            write_dates_schedule_atomic(schedule_path, initial_rows)
            logging.info("Created %s with %s date(s)", schedule_path, len(initial_rows))

        schedule_rows = [dict(r) for r in read_dates_schedule(schedule_path)]
        if not schedule_rows:
            logging.error("No rows in %s", schedule_path)
            return 2
        _mark_weekend_rows_skipped(schedule_path, schedule_rows)
        dates = pending_schedule_dates(schedule_rows)
        langfuse_dates_total = len(schedule_rows)
        if not dates:
            logging.error(
                "No pending weekdays in %s (all processed or weekends skipped?)",
                schedule_path,
            )
            return 2
        logging.info("Running %s pending date(s) from %s", len(dates), schedule_path)
    else:
        weekend_only = [d for d in bootstrap_dates if _is_weekend_ymd(d)]
        if weekend_only:
            logging.info(
                "Skipping weekend dates (no --dates-csv): %s",
                ", ".join(weekend_only),
            )
        dates = [d for d in bootstrap_dates if not _is_weekend_ymd(d)]
        if not dates:
            if bootstrap_dates:
                logging.error("All requested dates are weekends; nothing to run.")
            else:
                logging.error(
                    "Provide --dates and/or --dates-file, or --dates-csv with a non-empty schedule"
                )
            return 2

    config = _build_config()
    cost_bps = float(config.get("backtest_cost_bps", 0) or 0)
    if args.cost_bps is not None:
        cost_bps = float(args.cost_bps)

    if schedule_rows is not None and schedule_path is not None:
        seed_ledger, seed_last_close = last_successful_ledger_state(
            schedule_rows,
            initial_cash=args.initial_cash,
            cost_bps=cost_bps,
        )
        effective_initial_cash = float(seed_ledger.cash)

    from tradingagents.backtest.runner import run_backtest_mvp
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    stats_handler = StatsCallbackHandler()
    langfuse_handler = None
    if get_langfuse_client() is not None:
        langfuse_handler = get_langfuse_handler()

    callbacks = [stats_handler]
    if langfuse_handler is not None:
        callbacks.append(langfuse_handler)

    graph = TradingAgentsGraph(debug=args.debug, config=config, callbacks=callbacks)

    ticker = args.ticker.strip()
    langfuse_meta = {
        "llm_provider": config.get("llm_provider"),
        "quick_think_llm": config.get("quick_think_llm"),
        "deep_think_llm": config.get("deep_think_llm"),
    }

    def _on_day_complete(
        date: str,
        signal: str,
        equity: float | None,
        error: str | None,
        close: float | None,
        cash: float | None,
        shares: float | None,
        analysis: dict[str, str] | None = None,
    ) -> None:
        if schedule_path is None or schedule_rows is None:
            return
        eq_s = f"{equity:.6f}" if equity is not None else ""
        err_s = (error or "").strip()
        processed = err_s == ""
        close_s = f"{close:.6f}" if close is not None else ""
        cash_s = f"{cash:.6f}" if cash is not None else ""
        shares_s = f"{shares:.6f}" if shares is not None else ""
        if analysis is None:
            analysis = empty_schedule_analysis_values()
        try:
            update_schedule_row(
                schedule_rows,
                date,
                processed=processed,
                final_signal=signal,
                equity=eq_s,
                error=err_s,
                close=close_s,
                cash=cash_s,
                shares=shares_s,
                analysis=analysis,
            )
            write_dates_schedule_atomic(schedule_path, schedule_rows)
        except ValueError as e:
            logging.warning("Schedule update skipped for %s: %s", date, e)

    out = run_backtest_mvp(
        graph,
        ticker,
        dates,
        initial_cash=effective_initial_cash,
        buy_fraction=args.buy_fraction,
        cost_bps=cost_bps,
        use_llm_signal=args.use_llm_signal,
        results_dir=Path(args.results_dir) if args.results_dir else None,
        use_live_portfolio=False,
        langfuse_meta=langfuse_meta,
        on_day_complete=_on_day_complete if schedule_path is not None else None,
        initial_ledger=seed_ledger if schedule_path is not None else None,
        initial_last_close=seed_last_close if schedule_path is not None else None,
        langfuse_dates_total=langfuse_dates_total,
    )
    s = out["summary"]
    ann = s.get("annualized_return")
    sh = s.get("sharpe_ratio")
    ann_s = f"{ann:.4f}" if ann is not None else "n/a"
    sh_s = f"{sh:.4f}" if sh is not None else "n/a"
    logging.info(
        "Done. total_return=%.4f ann_return=%s sharpe=%s max_drawdown=%.4f "
        "cost_bps=%s total_fees=%.2f final_equity=%.2f -> %s",
        s["total_return"],
        ann_s,
        sh_s,
        s["max_drawdown"],
        s.get("cost_bps", 0),
        s.get("total_transaction_costs", 0.0),
        s["final_equity"],
        s["output_dir"],
    )
    print(s["final_equity"])  # simple last line like main.py prints decision
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
