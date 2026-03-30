#!/usr/bin/env python3
"""
MVP historical backtest: run TradingAgentsGraph per date (no live portfolio),
execute paper trades at close, write eval_results/<ticker>/backtest_mvp_<id>/.

After each date, equity.csv, trades.csv, summary.json are refreshed. Optional ``--dates-csv``
tracks progress: columns date, processed, final_signal, equity, error (atomic rewrites).

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
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cli.stats_handler import StatsCallbackHandler
from tradingagents.backtest.dates_schedule import (
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
                }
                for d in bootstrap_dates
            ]
            write_dates_schedule_atomic(schedule_path, initial_rows)
            logging.info("Created %s with %s date(s)", schedule_path, len(initial_rows))

        schedule_rows = [dict(r) for r in read_dates_schedule(schedule_path)]
        if not schedule_rows:
            logging.error("No rows in %s", schedule_path)
            return 2
        dates = pending_schedule_dates(schedule_rows)
        langfuse_dates_total = len(schedule_rows)
        if not dates:
            logging.error("No pending dates in %s (all processed?)", schedule_path)
            return 2
        logging.info("Running %s pending date(s) from %s", len(dates), schedule_path)
    else:
        dates = bootstrap_dates
        if not dates:
            logging.error(
                "Provide --dates and/or --dates-file, or --dates-csv with a non-empty schedule"
            )
            return 2

    from tradingagents.backtest.runner import run_backtest_mvp
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    config = _build_config()
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
    ) -> None:
        if schedule_path is None or schedule_rows is None:
            return
        eq_s = f"{equity:.6f}" if equity is not None else ""
        try:
            update_schedule_row(
                schedule_rows,
                date,
                processed=True,
                final_signal=signal,
                equity=eq_s,
                error=(error or "").strip(),
            )
            write_dates_schedule_atomic(schedule_path, schedule_rows)
        except ValueError as e:
            logging.warning("Schedule update skipped for %s: %s", date, e)

    out = run_backtest_mvp(
        graph,
        ticker,
        dates,
        initial_cash=args.initial_cash,
        buy_fraction=args.buy_fraction,
        use_llm_signal=args.use_llm_signal,
        results_dir=Path(args.results_dir) if args.results_dir else None,
        use_live_portfolio=False,
        langfuse_meta=langfuse_meta,
        on_day_complete=_on_day_complete if schedule_path is not None else None,
        langfuse_dates_total=langfuse_dates_total,
    )
    s = out["summary"]
    logging.info(
        "Done. total_return=%.4f max_drawdown=%.4f final_equity=%.2f -> %s",
        s["total_return"],
        s["max_drawdown"],
        s["final_equity"],
        s["output_dir"],
    )
    print(s["final_equity"])  # simple last line like main.py prints decision
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
