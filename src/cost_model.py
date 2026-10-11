"""Cost, carbon and delivery model (R2, R3, R5).

Turns the four tables into a list of *options*: one per (order, eligible plant, production week).
Each option carries its landed cost, carbon, ship date, arrival date and lateness. The solver
only chooses among these options, so every number it reports is computed here, in code.

Formulas (per unit, then x qty):
    production  = unit_cost + energy_kwh_per_unit * energy_price_per_kwh   (energy price optional)
    landed      = (production + transport_cost_per_unit) * (1 + duty_pct / 100)
    carbon_kg   = energy_kwh_per_unit * carbon_kg_per_kwh[week] + carbon_kg_per_unit
    ship_date   = planning_start + 7*week + SHIP_OFFSET_DAYS
    arrival     = ship_date + transit_days ;  late_days = max(0, arrival - due)
"""
from __future__ import annotations

import math
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import pandas as pd

from src.config import settings
from src.database import read_table
from src.models import parse_date

CAPACITY_SCOPE = os.getenv("CAPACITY_SCOPE", "auto")  # auto | plant | product
SHARED = "*"  # pool key for a capacity shared by all products of a plant


@dataclass
class Option:
    order_id: str
    plant: str
    week: int            # 0-based production week
    pool: str            # capacity pool: SHARED or a product code
    qty: int
    cost: float          # landed cost for the whole order
    carbon_kg: float
    ship_date: date
    arrival: date
    late_days: int
    unit_cost_breakdown: dict[str, float] = field(default_factory=dict)

    @property
    def on_time(self) -> bool:
        return self.late_days == 0


@dataclass
class ProblemData:
    start: date
    n_weeks: int
    horizon_weeks: int                 # weeks up to the last due date
    orders: dict[str, dict]            # order_id -> {product, qty, destination, due}
    plants: list[str]
    options: dict[str, list[Option]]   # order_id -> options
    capacity: dict[tuple[str, str, int], float]       # (plant, pool, week) -> effective units
    base_capacity: dict[tuple[str, str, int], float]  # before capacity events
    ineligible: dict[str, str]         # order_id -> reason it has no option at all
    assumptions: list[str]
    limitations: list[str]
    events_applied: list[dict]

    def week_start(self, w: int) -> date:
        return self.start + timedelta(days=7 * w)


def _loss_fraction(v: float) -> float:
    """capacity_loss_pct may arrive as 30 or 0.30; both mean 30%."""
    v = float(v)
    return v / 100.0 if v > 1.0 else v


def normalise_scenario(scenario: dict | None) -> dict:
    """Accept several shapes and return {capacity_changes: [...], carbon_cut_pct, force, ignore_events}."""
    s = dict(scenario or {})
    changes = list(s.get("capacity_changes") or [])
    single = s.get("capacity_change") or (s if ("plant" in s and any(k in s for k in
              ("capacity_loss_pct", "loss_pct", "capacity_reduction_pct"))) else None)
    if single:
        changes.append(single)
    norm = []
    for c in changes:
        loss = c.get("capacity_loss_pct", c.get("loss_pct", c.get("capacity_reduction_pct")))
        if c.get("plant") is None or loss is None:
            continue
        norm.append({"plant": str(c["plant"]).upper().replace("PLANT", "").strip(),
                     "loss": _loss_fraction(loss),
                     "from_date": c.get("from_date"), "to_date": c.get("to_date"),
                     "weeks": c.get("weeks")})
    return {"capacity_changes": norm,
            "carbon_cut_pct": s.get("carbon_cut_pct"),
            "force": s.get("force"),
            "ignore_events": bool(s.get("ignore_events", False))}


def load_frames(conn: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    return {t: read_table(conn, t) for t in ("orders", "plants", "transport_duties", "capacity_events")}


def build_problem(frames: dict[str, pd.DataFrame], scenario: dict | None = None,
                  planning_start: str | None = None) -> ProblemData:
    """Build every option and the effective capacity for one scenario."""
    sc = normalise_scenario(scenario)
    start = parse_date(planning_start or settings.PLANNING_START)
    orders_df, plants_df = frames["orders"], frames["plants"]
    trans_df, events_df = frames["transport_duties"], frames["capacity_events"]
    assumptions: list[str] = [
        f"Planning starts {start.isoformat()}; an order built in week w ships on day "
        f"{settings.SHIP_OFFSET_DAYS} of that week (Monday = day 0).",
        "Import duty is charged on production + energy + transport value.",
        "Each order is built in exactly one plant in one week (orders are never split).",
    ]
    limitations: list[str] = ["Duty rates, transit days and costs are assumed constant for the period."]

    if orders_df.empty or plants_df.empty or trans_df.empty:
        raise ValueError("orders, plants and transport_duties must all be loaded before planning")

    orders = {}
    for r in orders_df.to_dict("records"):
        orders[str(r["order_id"])] = {"product": str(r["product"]), "qty": int(float(r["qty"])),
                                      "destination": str(r["destination"]), "due": parse_date(r["due"])}
    max_due = max(o["due"] for o in orders.values())
    horizon_weeks = max(1, math.ceil(((max_due - start).days + 1) / 7))
    n_weeks = horizon_weeks + settings.EXTRA_WEEKS

    # ---- plant parameters, optionally varying by week --------------------------------
    has_energy_price = "energy_price_per_kwh" in plants_df.columns and plants_df["energy_price_per_kwh"].notna().any()
    if not has_energy_price:
        assumptions.append("No energy_price_per_kwh column: energy cost is taken as included in unit_cost.")
    has_week = "week" in plants_df.columns and plants_df["week"].notna().any()
    if has_week:
        assumptions.append("plants.csv has a week column: carbon intensity (and other plant values) vary by week; "
                           "the latest earlier week is used when a week is missing.")

    params: dict[tuple[str, str], dict[int | None, dict]] = {}
    for r in plants_df.to_dict("records"):
        key = (str(r["plant"]), str(r["product"]))
        wk = None
        if has_week and r.get("week") is not None and not pd.isna(r.get("week")):
            wk = int(float(r["week"])) - 1  # file is 1-based
        params.setdefault(key, {})[wk] = r

    def param(plant: str, product: str, w: int) -> dict | None:
        rows = params.get((plant, product))
        if not rows:
            return None
        if w in rows:
            return rows[w]
        earlier = [k for k in rows if k is not None and k <= w]
        if earlier:
            return rows[max(earlier)]
        if None in rows:
            return rows[None]
        return rows[min(k for k in rows if k is not None)]

    plants = sorted({p for p, _ in params})

    # ---- capacity pools ---------------------------------------------------------------
    shared: dict[str, bool] = {}
    for p in plants:
        caps = plants_df[plants_df["plant"].astype(str) == p]["weekly_capacity"].astype(float)
        if CAPACITY_SCOPE == "plant":
            shared[p] = True
        elif CAPACITY_SCOPE == "product":
            shared[p] = False
        else:
            shared[p] = caps.nunique() == 1 and len(caps) >= 1
    if CAPACITY_SCOPE == "auto":
        assumptions.append("weekly_capacity is read as shared by all products of a plant when every product row "
                           "of that plant has the same value, otherwise as a per-product limit.")

    base_cap: dict[tuple[str, str, int], float] = {}
    for (p, prod), rows in params.items():
        for w in range(n_weeks):
            r = param(p, prod, w)
            pool = SHARED if shared[p] else prod
            base_cap[(p, pool, w)] = float(r["weekly_capacity"])  # shared pools: same value from any row

    # ---- capacity events (stored) + scenario changes ----------------------------------
    events: list[dict] = []
    if not sc["ignore_events"] and not events_df.empty:
        for r in events_df.to_dict("records"):
            events.append({"source": "capacity_events.csv", "event_id": r.get("event_id"), "plant": str(r["plant"]),
                           "loss": _loss_fraction(r["capacity_loss_pct"]),
                           "from": parse_date(r["from_date"]), "to": parse_date(r["to_date"])})
    for i, c in enumerate(sc["capacity_changes"]):
        if c.get("weeks"):
            wks = [int(w) - 1 for w in c["weeks"]]
            f, t = start + timedelta(days=7 * min(wks)), start + timedelta(days=7 * max(wks) + 6)
        else:
            f = parse_date(c["from_date"]) if c.get("from_date") else start
            t = parse_date(c["to_date"]) if c.get("to_date") else start + timedelta(days=7 * n_weeks - 1)
        events.append({"source": "scenario", "event_id": f"S{i + 1}", "plant": c["plant"], "loss": c["loss"],
                       "from": f, "to": t})
        if not c.get("weeks") and not c.get("from_date"):
            assumptions.append(f"Capacity change at plant {c['plant']} has no dates, so it is applied to the "
                               f"whole planning horizon.")

    cap = dict(base_cap)
    for (p, pool, w), v in base_cap.items():
        ws, we = start + timedelta(days=7 * w), start + timedelta(days=7 * w + 6)
        factor = 1.0
        for e in events:
            if e["plant"] != p:
                continue
            overlap = (min(we, e["to"]) - max(ws, e["from"])).days + 1
            if overlap > 0:
                factor *= 1.0 - e["loss"] * overlap / 7.0  # pro-rated by days in the week
        cap[(p, pool, w)] = max(0.0, math.floor(v * factor))

    # ---- transport lookup ---------------------------------------------------------------
    trans = {(str(r["plant"]), str(r["destination"])): r for r in trans_df.to_dict("records")}

    # ---- options ------------------------------------------------------------------------
    options: dict[str, list[Option]] = {}
    ineligible: dict[str, str] = {}
    for oid, o in orders.items():
        opts: list[Option] = []
        makers = [p for p in plants if (p, o["product"]) in params]
        if not makers:
            ineligible[oid] = f"no plant can make product {o['product']}"
        for p in makers:
            t = trans.get((p, o["destination"]))
            if t is None:
                continue
            pool = SHARED if shared[p] else o["product"]
            for w in range(n_weeks):
                r = param(p, o["product"], w)
                energy_price = float(r.get("energy_price_per_kwh") or 0) if has_energy_price else 0.0
                production = float(r["unit_cost"]) + float(r["energy_kwh_per_unit"]) * energy_price
                transport = float(t["transport_cost_per_unit"])
                duty_rate = float(t["duty_pct"]) / 100.0
                landed = (production + transport) * (1 + duty_rate)
                carbon = float(r["energy_kwh_per_unit"]) * float(r["carbon_kg_per_kwh"]) + float(t["carbon_kg_per_unit"])
                ship = start + timedelta(days=7 * w + settings.SHIP_OFFSET_DAYS)
                arrival = ship + timedelta(days=int(float(t["transit_days"])))
                late = max(0, (arrival - o["due"]).days)
                opts.append(Option(oid, p, w, pool, o["qty"], round(landed * o["qty"], 2),
                                   round(carbon * o["qty"], 3), ship, arrival, late,
                                   {"production": round(production, 4), "transport": transport,
                                    "duty": round((production + transport) * duty_rate, 4),
                                    "carbon_per_unit": round(carbon, 4)}))
        # Keep every on-time week plus the first two late weeks per plant: later weeks are never better.
        pruned: list[Option] = []
        for p in makers:
            mine = [o for o in opts if o.plant == p]
            pruned += [o for o in mine if o.on_time] + [o for o in mine if not o.on_time][:2]
        opts = pruned
        if makers and not opts:
            ineligible[oid] = f"no transport route from {makers} to {o['destination']}"
        options[oid] = opts

    return ProblemData(start, n_weeks, horizon_weeks, orders, plants, options, cap, base_cap, ineligible,
                       assumptions, limitations, events)


def problem_from_db(conn: sqlite3.Connection, scenario: dict | None = None,
                    planning_start: str | None = None) -> ProblemData:
    return build_problem(load_frames(conn), scenario, planning_start)


def option_table(pdata: ProblemData, order_id: str) -> list[dict[str, Any]]:
    """Best option per plant for one order (earliest on-time week, else earliest week)."""
    out = []
    by_plant: dict[str, list[Option]] = {}
    for op in pdata.options.get(order_id, []):
        by_plant.setdefault(op.plant, []).append(op)
    for p, ops in sorted(by_plant.items()):
        on_time = [o for o in ops if o.on_time]
        best = min(on_time or ops, key=lambda o: (o.late_days, o.week))
        out.append({"plant": p, "cost": best.cost, "carbon_kg": best.carbon_kg, "week": best.week + 1,
                    "arrival": best.arrival.isoformat(), "late_days": best.late_days,
                    "can_be_on_time": bool(on_time), **best.unit_cost_breakdown})
    return out
