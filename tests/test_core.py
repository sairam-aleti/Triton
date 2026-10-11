"""Unit tests for the key functions. Expected answers are worked out by hand in eval/datasets.py."""
from src import graph
from src.checker import validate_plan
from src.cost_model import build_problem, load_frames
from src.guardrails import check_numbers
from src.intent import rule_parse
from src.loader import ingest_csv
from src.solver import compare_plans, plan_assignment, solve

COST_ONLY = {"cost": 1.0, "carbon": 0.0}
CARBON_ONLY = {"cost": 0.0, "carbon": 1.0}


def _plan(conn, weights=None, **kw):
    frames = load_frames(conn)
    pdata = build_problem(frames, kw.pop("scenario", None))
    plan = solve(pdata, weights, **kw)
    return plan, validate_plan(frames, pdata, plan), pdata


def test_landed_cost_and_carbon_formulas(make_db):
    conn = make_db("duty_trap")
    pdata = build_problem(load_frames(conn))
    a_us = next(o for o in pdata.options["O-US"] if o.plant == "A")
    assert a_us.cost == round((10 + 1) * 1.5 * 100, 2)          # (unit + transport) x (1 + duty) x qty
    assert a_us.carbon_kg == round((2 * 0.8 + 0.5) * 100, 3)    # (kWh x kg/kWh + transport) x qty


def test_cheapest_plant_late_is_avoided(make_db):
    plan, check, _ = _plan(make_db("cheapest_late", n=3), COST_ONLY)
    assert all(a["plant"] == "B" and a["on_time"] for a in plan["allocation"])
    assert check["valid"]


def test_duty_flips_cheapest_plant(make_db):
    plan, _, _ = _plan(make_db("duty_trap"), COST_ONLY)
    got = plan_assignment(plan)
    assert got["O-US"] == "B" and got["O-IN"] == "A"


def test_single_plant_product(make_db):
    plan, check, _ = _plan(make_db("single_plant"), COST_ONLY)
    assert plan_assignment(plan)["O-Q"] == "C" and check["eligibility_ok"]


def test_green_plant_saturates_and_spills(make_db):
    plan, check, _ = _plan(make_db("green_small"), CARBON_ONLY)
    by_plant = plan["kpis"]["orders_by_plant"]
    week1_c = [a for a in plan["allocation"] if a["plant"] == "C" and a["week"] == 1]
    assert len(week1_c) == 1                      # only one 400-unit order fits C's 500/week
    assert by_plant["B"] >= 1 and check["capacity_ok"]


def test_weights_change_the_plan(make_db):
    conn = make_db("cheapest_late", n=2)
    cheap, _, _ = _plan(conn, COST_ONLY)
    green, _, _ = _plan(conn, CARBON_ONLY)
    assert plan_assignment(cheap) != plan_assignment(green)
    assert green["kpis"]["total_carbon_t"] < cheap["kpis"]["total_carbon_t"]


def test_capacity_loss_replan_moves_orders(make_db):
    conn = make_db("cheapest_late", n=3)
    base, _, _ = _plan(conn, COST_ONLY)
    new, check, _ = _plan(conn, COST_ONLY, scenario={"plant": "B", "capacity_loss_pct": 100},
                          baseline=plan_assignment(base))
    comp = compare_plans(base, new)
    assert comp["orders_moved"] == 3 and all(m["to"] != "B" for m in comp["moved_orders"])
    assert check["capacity_ok"]


def test_on_time_only_flags_impossible_order(make_db):
    plan, _, _ = _plan(make_db("impossible_date"), COST_ONLY, on_time_only=True)
    assert [u["order_id"] for u in plan["unserved"]] == ["O-X"]


def test_weekly_carbon_moves_production_week(make_db):
    plan, _, _ = _plan(make_db("weekly_carbon"), CARBON_ONLY)
    assert plan["allocation"][0]["week"] != 1


def test_checker_catches_overload(make_db):
    conn = make_db("over_capacity")
    frames = load_frames(conn)
    pdata = build_problem(frames)
    from src.solver import greedy_cheapest
    bad = greedy_cheapest(pdata)          # ignores capacity on purpose
    assert not validate_plan(frames, pdata, bad)["capacity_ok"]


def test_number_check_rejects_invented_numbers():
    facts = {"kpis": {"total_cost": 1843210.5, "total_carbon_t": 409.6, "late_orders": 2,
                      "plant_utilisation": {"A": 0.88}}}
    assert check_numbers("Cost 1.84M, carbon 410 t, 2 late, A at 88%.", facts)["ok"]
    bad = check_numbers("We saved 5M and cut carbon by 50%.", facts)
    assert not bad["ok"] and "5" in bad["unverified"]


def test_loader_rejects_bad_rows_and_accepts_aliases(tmp_path):
    from src.database import get_connection
    conn = get_connection(str(tmp_path / "x.db"))
    res = ingest_csv(b"order_id,product_id,quantity,destination,due_date\nO1,P,-5,EU,2026-11-01\nO2,P,5,EU,2026-11-01\n",
                     "orders", conn)
    assert res["rows_loaded"] == 1 and res["rows_rejected"] == 1
    res = ingest_csv(b"order_id,qty\nO1,5\n", "orders", conn)
    assert res["status"] == "rejected"


def test_intent_parser():
    assert rule_parse("Plant B loses 30% capacity. Re-plan.").capacity_changes[0].loss_pct == 30
    assert rule_parse("What does a 10% carbon cut cost?").carbon_cut_pct == 10
    assert rule_parse("Why was order 77 built in Plant C?").intent == "explain"
    assert rule_parse("Load Plant C beyond its capacity").asks_to_break_rule


def test_graph_refuses_ineligible_force(make_db):
    conn = make_db("single_plant")
    r = graph.run("Put order O-Q in Plant A", conn=conn)
    assert r["answer_type"] == "refused"


def test_graph_infeasible_and_clarification(make_db):
    conn = make_db("impossible_date")
    assert graph.run("What is the cheapest plan that is on time?", conn=conn)["answer_type"] == "infeasible"
    assert graph.run("Plant Z loses 20% capacity", conn=conn)["answer_type"] == "clarification_needed"


def test_api_health():
    from fastapi.testclient import TestClient
    from src.api import app
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
