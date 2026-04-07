"""CSV schedule for backtest dates: pending rows, progress columns, atomic save."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, List, Mapping, MutableMapping

SCHEDULE_FIELDNAMES = (
    "date",
    "processed",
    "final_signal",
    "equity",
    "error",
    "close",
    "cash",
    "shares",
)


def _cell_str(row: Mapping[str, Any], key: str) -> str:
    v = row.get(key)
    if v is None:
        return ""
    return str(v).strip()


def is_row_processed(value: object) -> bool:
    if value is None:
        return False
    s = str(value).strip().lower()
    if not s:
        return False
    return s in ("1", "true", "yes", "y")


def _parse_float(value: object) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def last_successful_ledger_state(
    rows: List[Mapping[str, str]],
    *,
    initial_cash: float,
) -> tuple["PaperLedger", float | None]:
    """
    Seed resume state from the last successful row in a state CSV.

    A row is considered successful when:
    - `processed` is true (see `is_row_processed`)
    - `error` is empty

    Returns:
        (ledger, last_close) where last_close can be None when not available.
    """
    # Local import to avoid any potential import cycles.
    from tradingagents.backtest.ledger import PaperLedger

    cash: float | None = None
    shares: float | None = None
    last_close: float | None = None

    for r in rows:
        if not is_row_processed(r.get("processed")):
            continue
        if _cell_str(r, "error"):
            continue

        cash = _parse_float(r.get("cash"))
        shares = _parse_float(r.get("shares"))
        last_close = _parse_float(r.get("close"))

    return (
        PaperLedger(
            cash=float(initial_cash) if cash is None else float(cash),
            shares=0.0 if shares is None else float(shares),
        ),
        last_close,
    )


def read_dates_schedule(path: Path) -> List[MutableMapping[str, str]]:
    """Read dates CSV; return list of dicts with keys in SCHEDULE_FIELDNAMES (others preserved)."""
    rows: List[MutableMapping[str, str]] = []
    if not path.is_file():
        return rows
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return rows
        base = list(SCHEDULE_FIELDNAMES)
        for raw in reader:
            row: dict[str, str] = {k: _cell_str(raw, k) for k in reader.fieldnames}
            for k in base:
                row.setdefault(k, _cell_str(row, k))
            rows.append(row)
    return rows


def pending_schedule_dates(rows: List[Mapping[str, str]]) -> List[str]:
    """Dates in row order that are not yet processed."""
    out: List[str] = []
    seen: set[str] = set()
    for r in rows:
        d = _cell_str(r, "date")
        if not d or is_row_processed(r.get("processed")):
            continue
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def update_schedule_row(
    rows: List[MutableMapping[str, str]],
    date: str,
    *,
    processed: bool,
    final_signal: str = "",
    equity: str = "",
    error: str = "",
    close: str | None = None,
    cash: str | None = None,
    shares: str | None = None,
) -> None:
    """Update first row matching ``date`` (strip-compared)."""
    key = date.strip()
    for r in rows:
        if _cell_str(r, "date") == key:
            r["processed"] = "true" if processed else ""
            r["final_signal"] = final_signal
            r["equity"] = equity
            r["error"] = error
            if close is not None:
                r["close"] = close
            if cash is not None:
                r["cash"] = cash
            if shares is not None:
                r["shares"] = shares
            return
    raise ValueError(f"schedule has no row for date {date!r}")


def write_dates_schedule_atomic(path: Path, rows: List[Mapping[str, str]]) -> None:
    """Write schedule CSV via tempfile + replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out_fields = list(SCHEDULE_FIELDNAMES)
    # preserve extra columns from first row if any
    extra: List[str] = []
    if rows:
        for k in rows[0].keys():
            if k not in out_fields and k not in extra:
                extra.append(k)
    fieldnames = out_fields + extra
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({fn: _cell_str(r, fn) for fn in fieldnames})
    os.replace(tmp, path)
