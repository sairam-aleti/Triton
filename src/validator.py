"""Dataset validation (Section 4.5). Runs before any analysis and after every ingest.

    python -m src.validator          # prints the report for the current database

Checks
  1. every order's product can be made in at least one plant
  2. every plant x destination pair used by orders has a transport row
  3. capacity is positive, in units per week
  4. due dates are after the planning start
  5. carbon intensity values are in kg CO2e/kWh (and transport in kg CO2e/unit) and in a plausible range
  6. row counts match the data card (data/data_card.json), when one is present
  +  capacity events refer to known plants and have a loss between 0 and 100%
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from src.config import settings
from src.database import get_connection, read_table
from src.models import parse_date

MAX_PLAUSIBLE_KG_PER_KWH = 2.0  # coal-heavy grids are ~1.0; above 2 is almost certainly a unit error


def _check(name: str, status: str, details: str) -> dict[str, str]:
    return {"check": name, "status": status, "details": details}


def validate_dataset(conn: sqlite3.Connection) -> dict[str, Any]:
    orders, plants = read_table(conn, "orders"), read_table(conn, "plants")
    trans, events = read_table(conn, "transport_duties"), read_table(conn, "capacity_events")
    checks: list[dict[str, str]] = []
    start = parse_date(settings.PLANNING_START)

    loaded = {t: len(df) for t, df in [("orders", orders), ("plants", plants),
                                        ("transport_duties", trans), ("capacity_events", events)]}
    if orders.empty or plants.empty or trans.empty:
        missing = [t for t in ("orders", "plants", "transport_duties") if loaded[t] == 0]
        checks.append(_check("tables loaded", "fail", f"missing: {missing}"))
        return {"valid": False, "row_counts": loaded, "checks": checks}

    # 1. eligibility
    makeable = set(plants["product"].astype(str))
    bad = orders[~orders["product"].astype(str).isin(makeable)]
    checks.append(_check("every order's product can be made somewhere",
                         "pass" if bad.empty else "fail",
                         "all products covered" if bad.empty else
                         f"{len(bad)} orders, products {sorted(bad['product'].astype(str).unique())[:10]}"))
    single = plants.groupby(plants["product"].astype(str))["plant"].nunique()
    single = sorted(single[single == 1].index)
    if single:
        checks.append(_check("single-plant products (edge case, handled)", "info", f"{single}"))

    # 2. routes
    pairs = {(str(p), str(d)) for p in plants["plant"].astype(str).unique() for d in orders["destination"].astype(str).unique()}
    have = set(zip(trans["plant"].astype(str), trans["destination"].astype(str)))
    missing = sorted(pairs - have)
    checks.append(_check("every plant x destination pair has a transport row",
                         "pass" if not missing else "fail",
                         f"{len(pairs)} pairs present" if not missing else f"{len(missing)} missing, e.g. {missing[:5]}"))

    # 3. capacity
    caps = plants["weekly_capacity"].astype(float)
    checks.append(_check("capacity is positive (units per week)", "pass" if (caps > 0).all() else "fail",
                         f"min {caps.min():.0f}, max {caps.max():.0f} units/week"))

    # 4. due dates
    dues = orders["due"].map(parse_date)
    early = int((dues <= start).sum())
    checks.append(_check(f"due dates after planning start {start}", "pass" if early == 0 else "fail",
                         f"range {dues.min()} .. {dues.max()}" if early == 0 else f"{early} orders due on/before start"))

    # 5. carbon units
    ci = plants["carbon_kg_per_kwh"].astype(float)
    tc = trans["carbon_kg_per_unit"].astype(float)
    ok = ci.between(0, MAX_PLAUSIBLE_KG_PER_KWH).all() and (tc >= 0).all()
    checks.append(_check("carbon intensity in kg CO2e/kWh, transport carbon in kg CO2e/unit",
                         "pass" if ok else "fail",
                         f"intensity {ci.min():.3f}..{ci.max():.3f} kg/kWh; transport {tc.min():.2f}..{tc.max():.2f} kg/unit"))
    duty = trans["duty_pct"].astype(float)
    if duty.max() <= 1 and duty.max() > 0:
        checks.append(_check("duty_pct looks like a fraction", "warn",
                             "all duty values <= 1: read as percent (0.5 = 0.5%). Check the unit."))

    # 6. data card
    card_path = Path(settings.DATA_DIR) / "data_card.json"
    if card_path.exists():
        try:
            expected = json.loads(card_path.read_text())["row_counts"]
            diff = {t: (loaded.get(t), n) for t, n in expected.items() if loaded.get(t) != n}
            checks.append(_check("row counts match data card", "pass" if not diff else "warn",
                                 f"{loaded}" if not diff else f"loaded vs card: {diff} (different dataset loaded?)"))
        except Exception as exc:
            checks.append(_check("row counts match data card", "warn", f"could not read data card: {exc}"))
    else:
        checks.append(_check("row counts", "info", f"{loaded} (no data_card.json to compare)"))

    # + events
    if not events.empty:
        unknown = sorted(set(events["plant"].astype(str)) - set(plants["plant"].astype(str)))
        loss = events["capacity_loss_pct"].astype(float)
        good = not unknown and loss.between(0, 100).all()
        checks.append(_check("capacity events refer to known plants", "pass" if good else "fail",
                             f"{len(events)} events" if good else f"unknown plants {unknown}"))

    valid = all(c["status"] != "fail" for c in checks)
    return {"valid": valid, "row_counts": loaded, "planning_start": start.isoformat(), "checks": checks}


def print_report(report: dict[str, Any]) -> None:
    print(f"\nDATA VALIDATION  (planning start {report.get('planning_start')})  rows: {report['row_counts']}")
    for c in report["checks"]:
        print(f"  [{c['status'].upper():4}] {c['check']}: {c['details']}")
    print(f"  => {'VALID' if report['valid'] else 'INVALID'}\n")


if __name__ == "__main__":
    rep = validate_dataset(get_connection())
    print_report(rep)
    sys.exit(0 if rep["valid"] else 1)
