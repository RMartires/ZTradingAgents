import unittest
from datetime import datetime
from unittest.mock import patch


class FakeKite:
    def __init__(self, historical_records=None, ltp_price=2500.0, quote_payload=None):
        self._historical_records = historical_records or []
        self._ltp_price = ltp_price
        self._quote_payload = quote_payload or {
            "last_price": ltp_price,
            "volume": 1000,
            "ohlc": {"open": 2400.0, "high": 2550.0, "low": 2390.0, "close": 2450.0},
        }

    def historical_data(self, instrument_token, from_date, to_date, interval="day"):
        # Tests don't validate date filtering; just return provided records.
        return self._historical_records

    def ltp(self, kite_name: str):
        return {kite_name: {"last_price": self._ltp_price}}

    def quote(self, kite_name: str):
        return {kite_name: dict(self._quote_payload)}

    def instruments(self, exchange: str):
        # Not used in these tests because we patch the mapper.
        return []

    def holdings(self):
        return []

    def positions(self):
        return {"day": [], "net": []}

    def margins(self):
        return {}


class FakeMapper:
    def resolve(self, symbol: str, exchange: str = "NSE"):
        return {
            "instrument_token": 123,
            "tradingsymbol": symbol.split(".")[0].upper(),
            "exchange": exchange,
        }


class FakeSession:
    def __init__(self, kite: FakeKite):
        self._kite = kite

    def get_client(self):
        return self._kite


class TestKiteDataFunctions(unittest.TestCase):
    @patch("tradingagents.dataflows.kite_stock.get_instrument_mapper")
    @patch("tradingagents.dataflows.kite_stock.get_kite_session")
    def test_get_stock_data_format(self, mock_get_session, mock_get_mapper):
        from tradingagents.dataflows.kite_stock import get_stock_data

        records = [
            {
                "date": datetime(2024, 1, 1),
                "open": 2400.0,
                "high": 2450.0,
                "low": 2390.0,
                "close": 2440.0,
                "volume": 10,
            },
            {
                "date": datetime(2024, 1, 2),
                "open": 2440.0,
                "high": 2500.0,
                "low": 2430.0,
                "close": 2480.0,
                "volume": 20,
            },
        ]

        fake_kite = FakeKite(historical_records=records)
        mock_get_mapper.return_value = FakeMapper()
        mock_get_session.return_value = FakeSession(fake_kite)

        out = get_stock_data("RELIANCE.NS", "2024-01-01", "2024-01-31")

        self.assertIn("# Stock data for RELIANCE.NS from 2024-01-01 to 2024-01-31", out)
        self.assertIn("Date,Open,High,Low,Close,Adj Close,Volume", out)
        self.assertIn("2024-01-01,2400.0,2450.0,2390.0,2440.0,2440.0", out)

    @patch("tradingagents.dataflows.kite_stock.get_instrument_mapper")
    @patch("tradingagents.dataflows.kite_stock.get_kite_session")
    def test_get_ltp_format(self, mock_get_session, mock_get_mapper):
        from tradingagents.dataflows.kite_stock import get_ltp

        fake_kite = FakeKite(ltp_price=1234.5)
        mock_get_mapper.return_value = FakeMapper()
        mock_get_session.return_value = FakeSession(fake_kite)

        out = get_ltp("RELIANCE.NS")
        self.assertIn("## LTP for RELIANCE.NS", out)
        self.assertIn("1234.5", out)

    @patch("tradingagents.dataflows.kite_stock.get_instrument_mapper")
    @patch("tradingagents.dataflows.kite_stock.get_kite_session")
    def test_get_quote_returns_json(self, mock_get_session, mock_get_mapper):
        from tradingagents.dataflows.kite_stock import get_quote

        fake_kite = FakeKite(ltp_price=999.0)
        mock_get_mapper.return_value = FakeMapper()
        mock_get_session.return_value = FakeSession(fake_kite)

        out = get_quote("RELIANCE.NS")
        self.assertTrue(out.strip().startswith("{"))
        self.assertIn('"last_price": 999.0', out)

    @patch("tradingagents.dataflows.kite_indicator._get_stockstats_indicator_bulk")
    def test_get_indicators_window_format(self, mock_bulk):
        from tradingagents.dataflows.kite_indicator import get_indicators

        mock_bulk.return_value = {
            "2024-01-01": "9.0",
            "2024-01-02": "10.0",
        }

        out = get_indicators("RELIANCE.NS", "rsi", "2024-01-02", look_back_days=1)
        self.assertIn("## rsi values from 2024-01-01 to 2024-01-02:", out)
        self.assertIn("2024-01-02: 10.0", out)
        self.assertIn("2024-01-01: 9.0", out)

    def test_get_indicators_unsupported_indicator(self):
        from tradingagents.dataflows.kite_indicator import get_indicators

        with self.assertRaises(ValueError):
            get_indicators("RELIANCE.NS", "not_a_real_indicator", "2024-01-02", look_back_days=1)


if __name__ == "__main__":
    unittest.main()

