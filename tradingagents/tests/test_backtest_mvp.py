import json
import tempfile
import unittest
from pathlib import Path

from tradingagents.backtest.dates_schedule import (
    pending_schedule_dates,
    read_dates_schedule,
    update_schedule_row,
    write_dates_schedule_atomic,
    last_successful_ledger_state,
)
from tradingagents.backtest.ledger import PaperLedger
from tradingagents.backtest.prices import parse_close_from_vendor_block
from tradingagents.backtest.runner import max_drawdown, write_backtest_mvp_artifacts
from tradingagents.backtest.signals import normalize_signal_heuristic, resolve_signal


class TestNormalizeSignalHeuristic(unittest.TestCase):
    def test_final_proposal_line(self):
        text = "Some analysis.\n\nFINAL TRANSACTION PROPOSAL: **BUY**"
        self.assertEqual(normalize_signal_heuristic(text), "BUY")

    def test_last_token_wins(self):
        text = "Bear says SELL. Bull says BUY. FINAL: HOLD"
        self.assertEqual(normalize_signal_heuristic(text), "HOLD")

    def test_empty(self):
        self.assertIsNone(normalize_signal_heuristic(""))
        self.assertIsNone(normalize_signal_heuristic("   "))


class TestResolveSignal(unittest.TestCase):
    def test_processed_fallback(self):
        self.assertEqual(
            resolve_signal("no keywords here", processed="SELL"),
            "SELL",
        )

    def test_default_hold(self):
        self.assertEqual(resolve_signal("nothing"), "HOLD")


class TestParseCloseFromVendorBlock(unittest.TestCase):
    def test_kite_style_csv(self):
        block = """# Stock data for X from 2024-05-10 to 2024-05-11
# Total records: 1

Date,Open,High,Low,Close,Adj Close,Volume
2024-05-10,10.0,11.0,9.0,10.5,10.5,1000
"""
        self.assertAlmostEqual(
            parse_close_from_vendor_block(block, "2024-05-10"),
            10.5,
        )

    def test_yfinance_index_column(self):
        block = """# Stock data

,Open,High,Low,Close,Adj Close,Volume
2024-05-10,10.0,11.0,9.0,10.5,10.5,1000
"""
        self.assertAlmostEqual(
            parse_close_from_vendor_block(block, "2024-05-10"),
            10.5,
        )

    def test_no_data(self):
        self.assertIsNone(
            parse_close_from_vendor_block("No data found for symbol 'X'", "2024-05-10")
        )


class TestMaxDrawdown(unittest.TestCase):
    def test_simple_peak_then_drop(self):
        self.assertAlmostEqual(max_drawdown([100.0, 120.0, 90.0]), 0.25)

    def test_empty(self):
        self.assertEqual(max_drawdown([]), 0.0)


class TestDatesSchedule(unittest.TestCase):
    def test_pending_and_update_atomic(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "d.csv"
            rows = [
                {
                    "date": "2024-05-01",
                    "processed": "",
                    "final_signal": "",
                    "equity": "",
                    "error": "",
                },
                {
                    "date": "2024-05-02",
                    "processed": "true",
                    "final_signal": "BUY",
                    "equity": "",
                    "error": "",
                },
            ]
            write_dates_schedule_atomic(p, rows)
            loaded = read_dates_schedule(p)
            self.assertEqual(pending_schedule_dates(loaded), ["2024-05-01"])
            update_schedule_row(
                loaded,
                "2024-05-01",
                processed=True,
                final_signal="HOLD",
                equity="100000.000000",
                error="",
            )
            write_dates_schedule_atomic(p, loaded)
            again = read_dates_schedule(p)
            self.assertEqual(pending_schedule_dates(again), [])

    def test_pending_after_error_when_unprocessed(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "d.csv"
            rows = [
                {
                    "date": "2024-05-01",
                    "processed": "",
                    "final_signal": "",
                    "equity": "",
                    "error": "BadRequestError: Provider returned error",
                }
            ]
            write_dates_schedule_atomic(p, rows)
            loaded = read_dates_schedule(p)
            self.assertEqual(pending_schedule_dates(loaded), ["2024-05-01"])


class TestStateCSVResumeSeeding(unittest.TestCase):
    def test_last_successful_ledger_state(self):
        rows = [
            {
                "date": "2024-05-01",
                "processed": "true",
                "error": "",
                "cash": "1000",
                "shares": "2",
                "close": "50",
            },
            {
                "date": "2024-05-02",
                "processed": "true",
                "error": "BadRequestError: something went wrong",
                "cash": "2000",
                "shares": "3",
                "close": "60",
            },
            {
                "date": "2024-05-03",
                "processed": "",
                "error": "",
                "cash": "3000",
                "shares": "4",
                "close": "70",
            },
        ]

        ledger, last_close = last_successful_ledger_state(
            rows,
            initial_cash=9999.0,
        )

        self.assertAlmostEqual(ledger.cash, 1000.0)
        self.assertAlmostEqual(ledger.shares, 2.0)
        self.assertAlmostEqual(last_close or 0.0, 50.0)

    def test_last_successful_defaults_when_missing(self):
        rows = [
            {
                "date": "2024-05-01",
                "processed": "true",
                "error": "",
                "cash": "",
                "shares": "",
                "close": "",
            }
        ]
        ledger, last_close = last_successful_ledger_state(
            rows,
            initial_cash=1234.0,
        )
        self.assertAlmostEqual(ledger.cash, 1234.0)
        self.assertAlmostEqual(ledger.shares, 0.0)
        self.assertIsNone(last_close)


class TestWriteBacktestMvpArtifacts(unittest.TestCase):
    def test_writes_summary_status(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            ledger = PaperLedger(cash=100_000.0)
            rows = [
                {
                    "date": "2024-01-01",
                    "signal": "HOLD",
                    "close": 10.0,
                    "cash": 100_000.0,
                    "shares": 0.0,
                    "equity": 100_000.0,
                    "processed_signal": "",
                }
            ]
            s = write_backtest_mvp_artifacts(
                base,
                "TEST",
                "runid",
                100_000.0,
                2,
                rows,
                ledger,
                complete=False,
                last_completed="2024-01-01",
            )
            self.assertEqual(s["status"], "running")
            self.assertEqual(s["dates_completed"], 1)
            data = json.loads((base / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(data["last_completed_date"], "2024-01-01")
            self.assertTrue((base / "equity.csv").is_file())
            self.assertTrue((base / "trades.csv").is_file())

    def test_write_equity_trades_off(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            ledger = PaperLedger(cash=100_000.0)
            rows = [
                {
                    "date": "2024-01-01",
                    "signal": "HOLD",
                    "close": 10.0,
                    "cash": 100_000.0,
                    "shares": 0.0,
                    "equity": 100_000.0,
                    "processed_signal": "",
                }
            ]
            s = write_backtest_mvp_artifacts(
                base,
                "TEST",
                "runid",
                100_000.0,
                2,
                rows,
                ledger,
                complete=False,
                last_completed="2024-01-01",
                write_equity_trades=False,
            )
            self.assertEqual(s["status"], "running")
            self.assertTrue((base / "summary.json").is_file())
            self.assertFalse((base / "equity.csv").exists())
            self.assertFalse((base / "trades.csv").exists())


class TestPaperLedger(unittest.TestCase):
    def test_buy_then_sell(self):
        L = PaperLedger(cash=10_000.0)
        L.apply_signal("BUY", 100.0, buy_fraction=1.0, asof_date="2024-01-01")
        self.assertAlmostEqual(L.shares, 100.0)
        self.assertAlmostEqual(L.cash, 0.0)
        L.apply_signal("SELL", 110.0, asof_date="2024-01-02")
        self.assertAlmostEqual(L.shares, 0.0)
        self.assertAlmostEqual(L.cash, 11_000.0)

    def test_hold_no_change(self):
        L = PaperLedger(cash=5000.0, shares=10.0)
        L.apply_signal("HOLD", 100.0, asof_date="2024-01-01")
        self.assertAlmostEqual(L.cash, 5000.0)
        self.assertAlmostEqual(L.shares, 10.0)

    def test_equity(self):
        L = PaperLedger(cash=0.0, shares=2.0)
        self.assertAlmostEqual(L.equity(50.0), 100.0)


if __name__ == "__main__":
    unittest.main()
