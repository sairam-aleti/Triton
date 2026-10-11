"""Tiny hand-built datasets whose correct answer can be worked out by hand.

Each function returns {table: csv_text}. Dates are offsets from the planning start, so the
expected answers hold for any PLANNING_START. Used by the pytest suite and the evaluation runner.
"""
from __future__ import annotations

from datetime import date, timedelta

from src.config import settings

START = date.fromisoformat(settings.PLANNING_START)


def d(days: int) -> str:
    return (START + timedelta(days=days)).isoformat()


PLANTS_ABC = """plant,product,weekly_capacity,unit_cost,energy_kwh_per_unit,carbon_kg_per_kwh
A,P,1000,10,2,0.8
B,P,1000,12,2,0.4
C,P,1000,14,2,0.1
"""


def routes(rows: dict[tuple[str, str], tuple[float, int, float, float]]) -> str:
    out = "plant,destination,transport_cost_per_unit,transit_days,duty_pct,carbon_kg_per_unit\n"
    for (p, dest), (c, t, duty, kg) in rows.items():
        out += f"{p},{dest},{c},{t},{duty},{kg}\n"
    return out


NO_EVENTS = "event_id,plant,from_date,to_date,capacity_loss_pct\n"


def cheapest_late(n: int = 1) -> dict[str, str]:
    """Plant A is cheapest to EU but needs 20 days; orders are due in 12 days. Expect B (next cheapest on time)."""
    orders = "order_id,product,qty,destination,due\n" + "".join(
        f"O-{i},P,100,EU,{d(12 + i)}\n" for i in range(1, n + 1))
    r = routes({("A", "EU"): (1, 20, 0, 1.0), ("B", "EU"): (2, 3, 0, 0.2), ("C", "EU"): (2, 4, 0, 0.2)})
    return {"orders": orders, "plants": PLANTS_ABC, "transport_duties": r, "capacity_events": NO_EVENTS}


def duty_trap() -> dict[str, str]:
    """Plant A is cheapest at the gate but pays 50% duty into US. Expect B for US, A for IN."""
    orders = f"order_id,product,qty,destination,due\nO-US,P,100,US,{d(30)}\nO-IN,P,100,IN,{d(30)}\n"
    r = routes({("A", "US"): (1, 5, 50, 0.5), ("B", "US"): (1, 5, 0, 0.5), ("C", "US"): (1, 5, 0, 0.5),
                ("A", "IN"): (1, 2, 0, 0.1), ("B", "IN"): (1, 8, 10, 0.8), ("C", "IN"): (1, 8, 10, 0.8)})
    return {"orders": orders, "plants": PLANTS_ABC, "transport_duties": r, "capacity_events": NO_EVENTS}


def single_plant() -> dict[str, str]:
    """Product Q is made only by plant C, although C is the most expensive. Expect C."""
    plants = PLANTS_ABC + "C,Q,1000,14,2,0.1\n"
    orders = f"order_id,product,qty,destination,due\nO-Q,Q,100,EU,{d(30)}\nO-P,P,100,EU,{d(30)}\n"
    r = routes({(p, "EU"): (1, 3, 0, 0.2) for p in "ABC"})
    return {"orders": orders, "plants": plants, "transport_duties": r, "capacity_events": NO_EVENTS}


def green_small(qty: int = 400) -> dict[str, str]:
    """C is greenest but holds only 500/week. With carbon weight 1, C fills to capacity and the rest spills."""
    plants = PLANTS_ABC.replace("C,P,1000", "C,P,500")
    orders = "order_id,product,qty,destination,due\n" + "".join(
        f"O-{i},P,{qty},EU,{d(6)}\n" for i in range(1, 4))
    r = routes({(p, "EU"): (1, 1, 0, 0.1) for p in "ABC"})
    return {"orders": orders, "plants": plants, "transport_duties": r, "capacity_events": NO_EVENTS}


def over_capacity() -> dict[str, str]:
    """Total demand 4500 in week 1 vs 3000 capacity, with due dates that allow nothing later. Expect unserved/late."""
    orders = "order_id,product,qty,destination,due\n" + "".join(
        f"O-{i},P,900,EU,{d(6)}\n" for i in range(1, 6))
    r = routes({(p, "EU"): (1, 1, 0, 0.1) for p in "ABC"})
    plants = PLANTS_ABC.replace("1000", "1000")
    return {"orders": orders, "plants": plants, "transport_duties": r, "capacity_events": NO_EVENTS}


def impossible_date() -> dict[str, str]:
    """Due in 2 days but every route needs 5+ days after shipping. No plan can be on time."""
    orders = f"order_id,product,qty,destination,due\nO-X,P,100,EU,{d(2)}\nO-Y,P,100,EU,{d(30)}\n"
    r = routes({(p, "EU"): (1, 5, 0, 0.1) for p in "ABC"})
    return {"orders": orders, "plants": PLANTS_ABC, "transport_duties": r, "capacity_events": NO_EVENTS}


def weekly_carbon() -> dict[str, str]:
    """Plant A's grid is dirty in week 1 only. A carbon-weighted plan should build A's order in week 2."""
    plants = ("plant,product,weekly_capacity,unit_cost,energy_kwh_per_unit,carbon_kg_per_kwh,week\n"
              "A,P,1000,10,2,0.1,\nA,P,1000,10,2,2.0,1\nA,P,1000,10,2,0.1,2\n")
    orders = f"order_id,product,qty,destination,due\nO-W,P,100,EU,{d(20)}\n"
    r = routes({("A", "EU"): (1, 1, 0, 0.1)})
    return {"orders": orders, "plants": plants, "transport_duties": r, "capacity_events": NO_EVENTS}


DATASETS = {f.__name__: f for f in (cheapest_late, duty_trap, single_plant, green_small, over_capacity,
                                    impossible_date, weekly_carbon)}


def load_into(conn, data: dict[str, str]) -> None:
    from src.loader import ingest_csv
    for table in ("orders", "plants", "transport_duties", "capacity_events"):
        if table in data:
            res = ingest_csv(data[table].encode(), table, conn)
            if res["status"] == "rejected" and table != "capacity_events":
                raise ValueError(f"{table}: {res['errors']}")
