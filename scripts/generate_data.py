"""Synthetic data generator (Annex D). Seeded, so the same command always gives the same files.

    python scripts/generate_data.py --out data --seed 42 --start 2026-10-12

Plants (fictional):  A = low cost, carbon-heavy grid, large capacity, far from Europe
                     B = mid cost, Europe, only maker of product P4
                     C = hydro-powered, lowest carbon, most expensive, small capacity
Every edge case in Section 4.4 is planted on purpose; see DATA_CARD.md.
No real company or personal data is used.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import date, timedelta
from pathlib import Path

DESTS = ["Germany", "France", "UK", "Poland", "Italy", "Spain", "USA", "Canada", "Mexico", "India", "Japan", "Brazil"]
EU = {"Germany", "France", "UK", "Poland", "Italy", "Spain"}

PLANTS = {  # unit_cost, energy_price_per_kwh, carbon_kg_per_kwh, weekly_capacity
    "A": (20.0, 0.12, 0.80, 4000),
    "B": (24.0, 0.10, 0.45, 2800),
    "C": (28.0, 0.15, 0.08, 1500),
}
PRODUCTS = {"P1": 4.0, "P2": 5.0, "P3": 6.5, "P4": 5.5}  # energy kWh per unit
MAKERS = {"P1": "ABC", "P2": "ABC", "P3": "AC", "P4": "B"}  # P4: only Plant B can make it

# (transport cost/unit, transit days, duty %) per plant and destination group
ROUTES = {
    "A": {"EU": (3.0, 20, 4.0), "USA": (4.0, 22, 25.0), "Canada": (4.0, 21, 6.0), "Mexico": (4.5, 24, 8.0),
          "India": (1.0, 2, 0.0), "Japan": (3.0, 8, 5.0), "Brazil": (5.0, 26, 12.0)},
    "B": {"EU": (2.0, 3, 0.0), "USA": (5.0, 12, 3.0), "Canada": (5.0, 11, 3.0), "Mexico": (5.5, 14, 6.0),
          "India": (6.0, 20, 10.0), "Japan": (7.0, 22, 6.0), "Brazil": (6.0, 16, 10.0)},
    "C": {"EU": (3.0, 5, 0.0), "USA": (5.0, 10, 3.0), "Canada": (4.5, 9, 3.0), "Mexico": (6.0, 13, 6.0),
          "India": (7.0, 22, 10.0), "Japan": (8.0, 24, 6.0), "Brazil": (6.5, 15, 10.0)},
}
TRANSPORT_KG_PER_UNIT_PER_DAY = 0.12


def route(plant: str, dest: str) -> tuple[float, int, float]:
    key = "EU" if dest in EU else dest
    cost, days, duty = ROUTES[plant][key]
    if dest == "UK" and plant != "A":
        duty = 2.0  # small post-border duty from the EU plants
    return cost, days, duty


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", default="2026-10-12")
    ap.add_argument("--orders", type=int, default=150)
    args = ap.parse_args()
    rnd = random.Random(args.seed)
    start = date.fromisoformat(args.start)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- plants.csv (one row per plant x product; plant A has a week-2 carbon spike row) ----
    plants = []
    for p, (uc, price, ci, cap) in PLANTS.items():
        for prod, kwh in PRODUCTS.items():
            if p in MAKERS[prod]:
                plants.append({"plant": p, "product": prod, "weekly_capacity": cap, "unit_cost": uc,
                               "energy_kwh_per_unit": kwh, "carbon_kg_per_kwh": ci,
                               "energy_price_per_kwh": price, "week": ""})
                if p == "A":  # TRAP: carbon intensity changes by week (grid spike in week 2)
                    plants.append({"plant": p, "product": prod, "weekly_capacity": cap, "unit_cost": uc,
                                   "energy_kwh_per_unit": kwh, "carbon_kg_per_kwh": 1.10,
                                   "energy_price_per_kwh": price, "week": 2})
                    plants.append({"plant": p, "product": prod, "weekly_capacity": cap, "unit_cost": uc,
                                   "energy_kwh_per_unit": kwh, "carbon_kg_per_kwh": ci,
                                   "energy_price_per_kwh": price, "week": 3})

    # ---- transport_duties.csv (3 x 12) ----
    trans = []
    for p in PLANTS:
        for d in DESTS:
            cost, days, duty = route(p, d)
            trans.append({"plant": p, "destination": d, "transport_cost_per_unit": cost, "transit_days": days,
                          "duty_pct": duty, "carbon_kg_per_unit": round(days * TRANSPORT_KG_PER_UNIT_PER_DAY, 2)})

    # ---- orders.csv ----
    orders = []
    planted = [  # TRAP: cheapest plant (A) is late for early European orders
        ("P1", 400, "Germany", 12), ("P2", 350, "France", 14), ("P1", 300, "Poland", 13),
        ("P3", 250, "Italy", 15),
        # TRAP: duty makes the cheapest plant expensive for the USA
        ("P1", 300, "USA", 30), ("P2", 280, "USA", 32),
        # TRAP: only plant B can make P4; early due dates collide with the capacity event
        ("P4", 700, "Germany", 8), ("P4", 650, "France", 9), ("P4", 600, "UK", 10),
    ]
    for i, (prod, qty, dest, due_off) in enumerate(planted, start=1):
        orders.append({"order_id": f"O-{i:03d}", "product": prod, "qty": qty, "destination": dest,
                       "due": (start + timedelta(days=due_off)).isoformat()})
    for i in range(len(planted) + 1, args.orders + 1):
        prod = rnd.choices(list(PRODUCTS), weights=[40, 30, 20, 10])[0]
        dest = rnd.choice(DESTS)
        orders.append({"order_id": f"O-{i:03d}", "product": prod, "qty": rnd.randint(40, 260),
                       "destination": dest, "due": (start + timedelta(days=rnd.randint(14, 40))).isoformat()})

    # ---- capacity_events.csv: TRAP, capacity loss makes some orders impossible ----
    events = [{"event_id": "EV-01", "plant": "B", "from_date": start.isoformat(),
               "to_date": (start + timedelta(days=13)).isoformat(), "capacity_loss_pct": 60}]

    def write(name: str, rows: list[dict]) -> None:
        with open(out / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    write("orders", orders)
    write("plants", plants)
    write("transport_duties", trans)
    write("capacity_events", events)
    card = {"name": "synthetic-multiplant", "version": "1.0", "seed": args.seed, "planning_start": args.start,
            "row_counts": {"orders": len(orders), "plants": len(plants), "transport_duties": len(trans),
                           "capacity_events": len(events)}}
    (out / "data_card.json").write_text(json.dumps(card, indent=2))
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
