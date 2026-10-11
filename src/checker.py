"""Plan checker (the `checker` step in the trace). Annex A rules, re-checked independently.

The solver is trusted to optimise, not to be correct: this module re-reads the raw tables and
re-checks every assignment. It also labels the plans that are legitimately imperfect
(late or unserved orders), which is where `infeasible` answers come from.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from typing import Any

import pandas as pd

from src.config import settings
from src.cost_model import SHARED, ProblemData
from src.models import parse_date

TOL = 1e-6


def _plant_row(plants: pd.DataFrame, plant: str, product: str, week: int) -> dict | None:
    rows = plants[(plants["plant"].astype(str) == plant) & (plants["product"].astype(str) == product)]
    if rows.empty:
        return None
    if "week" in rows.columns and rows["week"].notna().any():
        # Same rule as cost_model.param: exact week, else latest earlier week, else the undated row.
        wk = rows[rows["week"].notna()].copy()
        wk["_w"] = wk["week"].astype(float).astype(int)
        exact = wk[wk["_w"] == week]
        if not exact.empty:
            return exact.iloc[0].to_dict()
        earlier = wk[wk["_w"] < week]
        if not earlier.empty:
            return earlier.loc[earlier["_w"].idxmax()].to_dict()
        undated = rows[rows["week"].isna()]
        if not undated.empty:
            return undated.iloc[0].to_dict()
        return wk.loc[wk["_w"].idxmin()].to_dict()
    return rows.iloc[0].to_dict()


def validate_plan(frames: dict[str, pd.DataFrame], pdata: ProblemData, plan: dict[str, Any]) -> dict[str, Any]:
    plants, trans, orders = frames["plants"], frames["transport_duties"], frames["orders"]
    order_rows = {str(r["order_id"]): r for r in orders.to_dict("records")}
    routes = {(str(r["plant"]), str(r["destination"])): r for r in trans.to_dict("records")}
    has_price = "energy_price_per_kwh" in plants.columns and plants["energy_price_per_kwh"].notna().any()

    violations: list[str] = []
    eligibility_ok = capacity_ok = assignment_ok = cost_ok = True
    used: dict[tuple[str, str, int], float] = defaultdict(float)
    late = []
    seen = set()

    for a in plan["allocation"]:
        oid, p, wk = a["order_id"], a["plant"], int(a["week"])
        if oid in seen:
            assignment_ok = False
            violations.append(f"{oid} assigned twice")
        seen.add(oid)
        o = order_rows.get(oid)
        if o is None:
            assignment_ok = False
            violations.append(f"{oid} is not a known order")
            continue
        prow = _plant_row(plants, p, str(o["product"]), wk)
        route = routes.get((p, str(o["destination"])))
        if prow is None or route is None:
            eligibility_ok = False
            violations.append(f"{oid}: plant {p} cannot make {o['product']} or has no route to {o['destination']}")
            continue
        pool = SHARED if (p, SHARED, wk - 1) in pdata.capacity else str(o["product"])
        used[(p, pool, wk - 1)] += int(float(o["qty"]))

        # Delivery: recompute arrival from the raw tables
        ship = pdata.start + timedelta(days=7 * (wk - 1) + settings.SHIP_OFFSET_DAYS)
        arrival = ship + timedelta(days=int(float(route["transit_days"])))
        due = parse_date(o["due"])
        if arrival.isoformat() != a["arrival"]:
            violations.append(f"{oid}: arrival mismatch {a['arrival']} vs {arrival}")
            assignment_ok = False
        if arrival > due:
            late.append({"order_id": oid, "plant": p, "arrival": arrival.isoformat(), "due": due.isoformat(),
                         "late_days": (arrival - due).days})

        # Landed cost: recompute and compare
        price = float(prow.get("energy_price_per_kwh") or 0) if has_price else 0.0
        unit = (float(prow["unit_cost"]) + float(prow["energy_kwh_per_unit"]) * price
                + float(route["transport_cost_per_unit"])) * (1 + float(route["duty_pct"]) / 100)
        if abs(unit * int(float(o["qty"])) - a["cost"]) > max(0.01, 1e-6 * a["cost"]):
            cost_ok = False
            violations.append(f"{oid}: cost mismatch {a['cost']} vs {unit * int(float(o['qty'])):.2f}")

    for key, q in used.items():
        cap = pdata.capacity.get(key, 0.0)
        if q > cap + TOL:
            capacity_ok = False
            violations.append(f"plant {key[0]} week {key[2] + 1}: {q:.0f} units > capacity {cap:.0f}")

    served = seen | {u["order_id"] for u in plan["unserved"]}
    missing = set(order_rows) - served
    if missing:
        assignment_ok = False
        violations.append(f"{len(missing)} orders neither planned nor flagged")

    return {"capacity_ok": capacity_ok, "eligibility_ok": eligibility_ok, "assignment_ok": assignment_ok,
            "cost_ok": cost_ok, "delivery_ok": not late, "late_orders": late,
            "unserved_orders": [u["order_id"] for u in plan["unserved"]],
            "violations": violations[:50], "valid": capacity_ok and eligibility_ok and assignment_ok and cost_ok}
