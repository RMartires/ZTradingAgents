"""CSV schedule for backtest dates: pending rows, progress columns, atomic save."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, List, Mapping, MutableMapping

SCHEDULE_FIELDNAMES = ("date", "processed", "final_signal", "equity", "error")


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
) -> None:
    """Update first row matching ``date`` (strip-compared)."""
    key = date.strip()
    for r in rows:
        if _cell_str(r, "date") == key:
            r["processed"] = "true" if processed else ""
            r["final_signal"] = final_signal
            r["equity"] = equity
            r["error"] = error
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
