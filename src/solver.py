"""Allocation solver (R4-R8). A mixed-integer program over the options built in cost_model.

Decision variables
    x[o,p,w] in {0,1}  order o is built at plant p in week w (one option from cost_model)
    u[o]     in {0,1}  order o is left unserved (only when capacity truly runs out)

Constraints
    sum_{p,w} x[o,p,w] + u[o] = 1                        every order is planned once or flagged
    sum_{o in pool} qty_o * x[o,p,w] <= capacity[p,pool,w] weekly capacity after capacity events
    (optional) sum carbon * x <= carbon_cap               epsilon-constraint for the trade-off view
    (optional) late options removed                       "cheapest plan that is on time"
    (optional) x restricted to one plant for an order     counterfactual / forced assignment

Back-ends: OR-Tools CP-SAT (default, SOLVER=cpsat) or CBC through PuLP (SOLVER=cbc). Same model.

Objective (all terms are dimensionless so the weights mean what they say)
    w_cost   * Cost   / C0      C0 = sum of each order's cheapest option
  + w_carbon * Carbon / E0      E0 = sum of each order's greenest option
  + w_late   * sum(late_score) / N     late_score = 1 + late_days/7 for a late option
  + UNSERVED_PENALTY * sum u           larger than any cost/carbon/lateness trade-off
  + CHURN_PENALTY / N * (orders moved off their baseline plant)    only when re-planning
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import pulp

from src.config import settings
from src.cost_model import Option, ProblemData

UNSERVED_PENALTY = 10.0   # per unserved order; dominates every other term on purpose
CHURN_PENALTY = 0.1       # x 1/N per moved order: moves only if it saves ~14% of an average order's cost
LATE_FIXED = 1.0          # a late order costs 1 unit plus 1/7 per late day (normalised by N)


def resolve_weights(weights: dict | None) -> dict[str, float]:
    """Fill defaults and make cost + carbon sum to 1."""
    w = {k: float(v) for k, v in (weights or {}).items() if v is not None}
    if "cost" in w and "carbon" in w:
        c, e = w["cost"], w["carbon"]
    elif "cost" in w:
        c, e = w["cost"], max(0.0, 1 - w["cost"])
    elif "carbon" in w:
        c, e = max(0.0, 1 - w["carbon"]), w["carbon"]
    else:
        c, e = settings.DEFAULT_W_COST, settings.DEFAULT_W_CARBON
    total = c + e
    if total <= 0:
        raise ValueError("cost and carbon weights cannot both be zero")
    return {"cost": round(c / total, 4), "carbon": round(e / total, 4),
            "lateness": float(w.get("lateness", settings.DEFAULT_W_LATENESS))}


def _late_score(op: Option) -> float:
    return 0.0 if op.on_time else LATE_FIXED + op.late_days / 7.0


def solve(pdata: ProblemData, weights: dict | None = None, *, on_time_only: bool = False,
          carbon_cap_kg: float | None = None, baseline: dict[str, str] | None = None,
          force: dict[str, str] | None = None, label: str = "plan",
          hint: set | None = None) -> dict[str, Any]:
    """Solve one allocation. Returns a plan dict (allocation, unserved, kpis, status...)."""
    t0 = time.perf_counter()
    w = resolve_weights(weights)
    n = max(1, len(pdata.orders))

    # Candidate options after the optional filters
    cand: dict[str, list[Option]] = {}
    for oid, ops in pdata.options.items():
        ops2 = ops
        if on_time_only:
            ops2 = [o for o in ops2 if o.on_time]
        if force and oid in force:
            ops2 = [o for o in ops2 if o.plant == force[oid]]
        cand[oid] = ops2

    # Normalisers from the unconstrained best per order (never zero)
    c0 = sum(min((o.cost for o in ops), default=0) for ops in pdata.options.values()) or 1.0
    e0 = sum(min((o.carbon_kg for o in ops), default=0) for ops in pdata.options.values()) or 1.0

    # One objective coefficient per option (dimensionless), shared by both solver back-ends
    opts = [op for ops in cand.values() for op in ops]
    coef = [w["cost"] * op.cost / c0 + w["carbon"] * op.carbon_kg / e0 + w["lateness"] * _late_score(op) / n
            for op in opts]
    churn_k = CHURN_PENALTY / n if baseline else 0.0
    if baseline:
        # moved = 1 - stay - unserved  ->  subtract churn_k from every option that keeps the baseline plant
        coef = [c - (churn_k if baseline.get(op.order_id) == op.plant else 0.0) for c, op in zip(coef, opts)]

    backend = (settings.SOLVER or "cpsat").lower()
    run = _solve_cpsat if backend == "cpsat" else _solve_cbc
    picked, status, objective, gap = run(pdata, opts, coef, carbon_cap_kg, label, hint)
    if baseline:
        objective += churn_k * sum(1 for oid in baseline if oid in pdata.orders)  # constant part of churn
    chosen = [opts[i] for i in picked]

    plan = build_plan(pdata, chosen, label=label, weights=w)
    plan["status"] = status
    plan["solver"] = backend
    plan["objective"] = round(objective, 6)
    plan["gap"] = gap
    plan["solve_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    plan["options_considered"] = len(opts)
    plan["constraints"] = {"on_time_only": on_time_only, "carbon_cap_kg": carbon_cap_kg,
                           "force": force or {}, "churn_penalty": bool(baseline)}
    return plan


def _solve_cpsat(pdata: ProblemData, opts: list[Option], coef: list[float],
                 carbon_cap_kg: float | None, label: str, hint: set | None = None) -> tuple[list[int], str, float, float]:
    """OR-Tools CP-SAT back-end (default). Objective scaled to integers (1e6 = one objective unit)."""
    from ortools.sat.python import cp_model

    m = cp_model.CpModel()
    x = [m.NewBoolVar(f"x{i}") for i in range(len(opts))]
    u = {oid: m.NewBoolVar(f"u_{oid}") for oid in pdata.orders}
    by_order: dict[str, list[int]] = defaultdict(list)
    by_pool: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for i, op in enumerate(opts):
        by_order[op.order_id].append(i)
        by_pool[(op.plant, op.pool, op.week)].append(i)
    for oid in pdata.orders:
        m.Add(sum(x[i] for i in by_order.get(oid, [])) + u[oid] == 1)
    for key, idx in by_pool.items():
        m.Add(sum(opts[i].qty * x[i] for i in idx) <= int(pdata.capacity.get(key, 0)))
    if carbon_cap_kg is not None:
        m.Add(sum(int(round(opts[i].carbon_kg * 1000)) * x[i] for i in range(len(opts)))
              <= int(carbon_cap_kg * 1000))
    scale = 1_000_000
    m.Minimize(sum(int(round(c * scale)) * x[i] for i, c in enumerate(coef))
               + int(UNSERVED_PENALTY * scale) * sum(u.values()))
    if hint:
        for i, op in enumerate(opts):
            m.AddHint(x[i], (op.order_id, op.plant, op.week) in hint)
    solver = cp_model.CpSolver()
    # Deterministic: interleaved workers + a deterministic time budget give the same plan for the same
    # input on any machine. The wall-clock limit is only a safety net.
    solver.parameters.num_workers = settings.SOLVER_WORKERS
    solver.parameters.interleave_search = True
    solver.parameters.random_seed = 0
    solver.parameters.max_deterministic_time = float(settings.SOLVER_DET_TIME)
    solver.parameters.max_time_in_seconds = float(settings.SOLVER_TIME_LIMIT_S)
    solver.parameters.relative_gap_limit = settings.SOLVER_GAP
    st = solver.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [], "INFEASIBLE" if st == cp_model.INFEASIBLE else "NOT_SOLVED", 0.0, 1.0
    picked = [i for i in range(len(opts)) if solver.Value(x[i])]
    obj, bound = solver.ObjectiveValue(), solver.BestObjectiveBound()
    gap = round(abs(obj - bound) / max(1.0, abs(obj)), 5)
    return picked, "OPTIMAL" if st == cp_model.OPTIMAL else "FEASIBLE", obj / scale, gap


def _solve_cbc(pdata: ProblemData, opts: list[Option], coef: list[float],
               carbon_cap_kg: float | None, label: str, hint: set | None = None) -> tuple[list[int], str, float, float]:
    """PuLP + CBC back-end (kept for comparison in the evaluation)."""
    prob = pulp.LpProblem(label.replace(" ", "_")[:40] or "plan", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(len(opts))]
    u = {oid: pulp.LpVariable(f"u_{j}", cat="Binary") for j, oid in enumerate(pdata.orders)}
    by_order: dict[str, list[int]] = defaultdict(list)
    by_pool: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for i, op in enumerate(opts):
        by_order[op.order_id].append(i)
        by_pool[(op.plant, op.pool, op.week)].append(i)
    prob += pulp.lpSum(c * x[i] for i, c in enumerate(coef)) + UNSERVED_PENALTY * pulp.lpSum(u.values())
    for oid in pdata.orders:
        prob += pulp.lpSum(x[i] for i in by_order.get(oid, [])) + u[oid] == 1
    for key, idx in by_pool.items():
        prob += pulp.lpSum(opts[i].qty * x[i] for i in idx) <= pdata.capacity.get(key, 0.0)
    if carbon_cap_kg is not None:
        prob += pulp.lpSum(opts[i].carbon_kg * x[i] for i in range(len(opts))) <= carbon_cap_kg
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=settings.SOLVER_TIME_LIMIT_S, gapRel=settings.SOLVER_GAP))
    if prob.sol_status not in (1, 2):
        return [], pulp.LpStatus[prob.status].upper(), 0.0, 1.0
    picked = [i for i, v in enumerate(x) if v.varValue is not None and v.varValue > 0.5]
    status = "OPTIMAL" if prob.sol_status == 1 and prob.status == 1 else "FEASIBLE"
    return picked, status, float(pulp.value(prob.objective) or 0.0), 0.0 if status == "OPTIMAL" else -1.0


def build_plan(pdata: ProblemData, chosen: list[Option], *, label: str, weights: dict) -> dict[str, Any]:
    """Turn chosen options into the response shape and compute KPIs."""
    by_order = {op.order_id: op for op in chosen}
    allocation, unserved = [], []
    for oid, o in pdata.orders.items():
        op = by_order.get(oid)
        if op is None:
            reason = pdata.ineligible.get(oid) or "not enough capacity in any eligible plant"
            unserved.append({"order_id": oid, "product": o["product"], "qty": o["qty"],
                             "destination": o["destination"], "due": o["due"].isoformat(), "reason": reason})
            continue
        allocation.append({"order_id": oid, "plant": op.plant, "week": op.week + 1,
                           "ship_date": op.ship_date.isoformat(), "arrival": op.arrival.isoformat(),
                           "due": o["due"].isoformat(), "on_time": op.on_time, "late_days": op.late_days,
                           "qty": op.qty, "cost": op.cost, "carbon_kg": op.carbon_kg})

    used: dict[tuple[str, str, int], float] = defaultdict(float)
    for op in chosen:
        used[(op.plant, op.pool, op.week)] += op.qty
    last_week = max([pdata.horizon_weeks - 1] + [op.week for op in chosen])
    util = {}
    for p in pdata.plants:
        cap = sum(v for (pp, _, w), v in pdata.capacity.items() if pp == p and w <= last_week)
        u_ = sum(v for (pp, _, w), v in used.items() if pp == p)
        util[p] = round(u_ / cap, 4) if cap else 0.0
    weekly = [{"plant": p, "pool": pool, "week": w + 1, "used": used.get((p, pool, w), 0.0), "capacity": cap}
              for (p, pool, w), cap in sorted(pdata.capacity.items()) if w <= last_week]

    total_cost = round(sum(a["cost"] for a in allocation), 2)
    total_carbon = sum(a["carbon_kg"] for a in allocation)
    kpis = {"total_cost": total_cost, "total_carbon_t": round(total_carbon / 1000, 3),
            "late_orders": sum(1 for a in allocation if not a["on_time"]),
            "unserved_orders": len(unserved), "orders_planned": len(allocation),
            "orders_total": len(pdata.orders), "plant_utilisation": util,
            "orders_by_plant": {p: sum(1 for a in allocation if a["plant"] == p) for p in pdata.plants}}
    return {"label": label, "weights": weights, "allocation": allocation, "unserved": unserved,
            "kpis": kpis, "weekly_utilisation": weekly}


def greedy_cheapest(pdata: ProblemData) -> dict[str, Any]:
    """Level-2 baseline: each order to its cheapest plant, earliest week, ignoring capacity and dates."""
    chosen = []
    for oid, ops in pdata.options.items():
        if ops:
            chosen.append(min(ops, key=lambda o: (o.cost, o.week)))
    plan = build_plan(pdata, chosen, label="greedy cheapest", weights={"cost": 1.0, "carbon": 0.0, "lateness": 0.0})
    plan["status"] = "HEURISTIC"
    return plan


def plan_assignment(plan: dict) -> dict[str, str]:
    return {a["order_id"]: a["plant"] for a in plan["allocation"]}


def compare_plans(base: dict, new: dict) -> dict[str, Any]:
    """Differences between two plans: KPI deltas and the list of orders that moved."""
    b, n = plan_assignment(base), plan_assignment(new)
    base_rows = {a["order_id"]: a for a in base["allocation"]}
    new_rows = {a["order_id"]: a for a in new["allocation"]}
    moved = []
    for oid in sorted(set(b) | set(n)):
        if b.get(oid) != n.get(oid):
            br, nr = base_rows.get(oid), new_rows.get(oid)
            moved.append({"order_id": oid, "from": b.get(oid) or "unserved", "to": n.get(oid) or "unserved",
                          "cost_delta": round((nr["cost"] if nr else 0) - (br["cost"] if br else 0), 2),
                          "carbon_delta_kg": round((nr["carbon_kg"] if nr else 0) - (br["carbon_kg"] if br else 0), 3),
                          "now_on_time": bool(nr and nr["on_time"])})
    kb, kn = base["kpis"], new["kpis"]
    total = max(1, len(base["allocation"]))
    unchanged = sum(1 for oid, p in b.items() if n.get(oid) == p)
    return {"base": base["label"], "new": new["label"],
            "cost_delta": round(kn["total_cost"] - kb["total_cost"], 2),
            "cost_delta_pct": round(100 * (kn["total_cost"] - kb["total_cost"]) / kb["total_cost"], 2) if kb["total_cost"] else 0.0,
            "carbon_delta_t": round(kn["total_carbon_t"] - kb["total_carbon_t"], 3),
            "late_orders_delta": kn["late_orders"] - kb["late_orders"],
            "unserved_delta": kn["unserved_orders"] - kb["unserved_orders"],
            "orders_moved": len(moved), "orders_unchanged_pct": round(100 * unchanged / total, 1),
            "moved_orders": moved}


def tradeoff_frontier(pdata: ProblemData, cuts: list[float] | None = None,
                      weights: dict | None = None) -> dict[str, Any]:
    """Epsilon-constraint frontier: minimise cost subject to carbon <= (1 - cut) * carbon of the cheapest plan."""
    cuts = sorted(set([0.0] + [float(c) for c in (cuts or [5, 10, 15, 20, 25, 30])]))
    lat = (weights or {}).get("lateness")
    base = solve(pdata, {"cost": 1.0, "carbon": 0.0, "lateness": lat}, label="cheapest (0% cut)")
    base_carbon_kg = base["kpis"]["total_carbon_t"] * 1000
    points, plans = [], {0.0: base}
    for cut in cuts:
        prev = plans[max(plans)] if plans else base
        hint = {(a["order_id"], a["plant"], a["week"] - 1) for a in prev["allocation"]}
        plan = base if cut == 0 else solve(pdata, {"cost": 1.0, "carbon": 0.0, "lateness": lat},
                                           carbon_cap_kg=base_carbon_kg * (1 - cut / 100.0), label=f"{cut:g}% carbon cut",
                                           hint=hint)
        k = plan["kpis"]
        feasible = plan["status"] in ("OPTIMAL", "FEASIBLE") and k["unserved_orders"] <= base["kpis"]["unserved_orders"]
        extra = round(k["total_cost"] - base["kpis"]["total_cost"], 2)
        saved_t = base["kpis"]["total_carbon_t"] - k["total_carbon_t"]
        points.append({"carbon_cut_pct": cut, "cost": k["total_cost"], "carbon_t": k["total_carbon_t"],
                       "late_orders": k["late_orders"], "unserved_orders": k["unserved_orders"],
                       "extra_cost": extra,
                       "extra_cost_pct": round(100 * extra / base["kpis"]["total_cost"], 2) if base["kpis"]["total_cost"] else 0.0,
                       "cost_per_t_saved": round(extra / saved_t, 2) if saved_t > 1e-9 else 0.0,
                       "feasible": feasible})
        plans[cut] = plan
        if not feasible:
            break  # deeper cuts are impossible without dropping orders
    return {"points": points, "plans": plans, "base": base}


def weight_sweep(pdata: ProblemData, steps: int = 11) -> list[dict[str, Any]]:
    """Plans for carbon weight 0.0..1.0. Used to defend the default weight (README / eval report)."""
    out = []
    for i in range(steps):
        wc = round(i / (steps - 1), 2)
        plan = solve(pdata, {"cost": 1 - wc, "carbon": wc}, label=f"w_carbon={wc}")
        k = plan["kpis"]
        out.append({"w_carbon": wc, "cost": k["total_cost"], "carbon_t": k["total_carbon_t"],
                    "late_orders": k["late_orders"], "unserved_orders": k["unserved_orders"]})
    return out
