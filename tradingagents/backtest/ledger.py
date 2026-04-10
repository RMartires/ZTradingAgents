from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal


Signal = Literal["BUY", "SELL", "HOLD"]


@dataclass
class TradeRecord:
    trade_date: str
    signal: str
    close_price: float
    shares_before: float
    shares_after: float
    cash_before: float
    cash_after: float
    fees_paid: float = 0.0


@dataclass
class PaperLedger:
    """Single-asset long-only paper book: BUY uses a fraction of cash; SELL flattens.

    ``cost_bps`` charges that many basis points on traded notional per BUY/SELL
    (HOLDS are free): fee = notional * (cost_bps / 10_000), paid in addition
    to the cash deployed on buys and deducted from proceeds on sells.
    """

    cash: float
    shares: float = 0.0
    trades: List[TradeRecord] = field(default_factory=list)
    cost_bps: float = 0.0

    def equity(self, close_price: float) -> float:
        return self.cash + self.shares * close_price

    def apply_signal(
        self,
        signal: str,
        close_price: float,
        buy_fraction: float = 1.0,
        asof_date: str = "",
    ) -> None:
        if close_price <= 0:
            return

        s = (signal or "").strip().upper()
        if s not in ("BUY", "SELL", "HOLD"):
            s = "HOLD"

        cash_before = self.cash
        shares_before = self.shares
        fees_paid = 0.0
        rate = max(0.0, float(self.cost_bps)) / 10_000.0

        if s == "HOLD":
            pass
        elif s == "SELL" and self.shares > 0:
            proceeds = self.shares * close_price
            fees_paid = proceeds * rate
            self.cash += proceeds - fees_paid
            self.shares = 0.0
        elif s == "BUY":
            frac = max(0.0, min(1.0, float(buy_fraction)))
            spend = self.cash * frac
            if spend > 0:
                fees_paid = spend * rate
                new_shares = spend / close_price
                self.cash -= spend + fees_paid
                self.shares += new_shares

        self.trades.append(
            TradeRecord(
                trade_date=asof_date,
                signal=s,
                close_price=close_price,
                shares_before=shares_before,
                shares_after=self.shares,
                cash_before=cash_before,
                cash_after=self.cash,
                fees_paid=fees_paid,
            )
        )
