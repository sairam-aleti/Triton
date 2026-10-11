"""LangGraph orchestration for POST /analyze.

    intake (parse) -> guard -> [optimiser | scenario agent | explain | brief | (exit)]
                   -> checker -> writer -> number check (retry writer once) -> respond

Only two steps use the LLM: intake (structured parsing) and the writer; the scenario agent uses
it to choose tool calls. Everything that produces a number is a deterministic tool.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from src import intent as intent_mod
from src.agents import writer
from src.agents.scenario import ScenarioRunner, run_agent
from src.checker import validate_plan
from src.config import settings
from src.cost_model import option_table
from src.database import get_connection, get_data_version, save_run
from src.guardrails import check_numbers
from src.planning import Planner
from src.solver import compare_plans, resolve_weights

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------------
# Run context (mutable, shared by nodes) and graph state
# --------------------------------------------------------------------------------------------
@dataclass
class RunContext:
    run_id: str
    request: str
    conn: Any
    planner: Optional[Planner] = None
    steps: list[dict] = field(default_factory=list)
    llm_calls: int = 0
    tokens: int = 0
    t0: float = field(default_factory=time.perf_counter)

    def record(self, agent: str, tool: str, input: Any = None, output: Any = None, duration_ms: float = 0.0,
               **extra: Any) -> None:
        self.steps.append({"agent": agent, "tool": tool, "input": _short(input), "output": _short(output),
                           "duration_ms": duration_ms, **extra})


class State(TypedDict, total=False):
    ctx: RunContext
    weights_in: Optional[dict]
    scenario_in: Optional[dict]
    parsed: dict
    intent: str
    decision: str            # proceed | refused | clarification_needed
    message: str
    weights: dict
    primary_plan: Optional[dict]
    base_plan: Optional[dict]
    comparison: Optional[dict]
    tradeoff: list
    explanation: Optional[dict]
    validator: Optional[dict]
    facts: dict
    answer_type: str
    text: str
    write_attempts: int
    feedback: list
    number_check: dict
    response: dict
    _scenario_for_primary: Optional[dict]


def _short(x: Any, limit: int = 1500) -> Any:
    """Keep traces readable: plans are summarised, long values truncated."""
    if isinstance(x, dict) and "allocation" in x and "kpis" in x:
        return {"label": x.get("label"), "status": x.get("status"), "gap": x.get("gap"),
                "solve_ms": x.get("solve_ms"), "kpis": x.get("kpis")}
    if isinstance(x, dict):
        return {k: _short(v, limit) for k, v in x.items()}
    if isinstance(x, list) and len(x) > 12:
        return [_short(v, limit) for v in x[:12]] + [f"... {len(x) - 12} more"]
    if isinstance(x, str) and len(x) > limit:
        return x[:limit] + "..."
    return x


def _timed(ctx: RunContext, agent: str, tool: str, fn, input: Any = None):
    t = time.perf_counter()
    out = fn()
    ctx.record(agent, tool, input, out, round((time.perf_counter() - t) * 1000, 1))
    return out


def _find_order(orders: dict, ref: Optional[str]) -> Optional[str]:
    if not ref:
        return None
    ref = ref.strip().upper()
    if ref in orders:
        return ref
    digits = "".join(ch for ch in ref if ch.isdigit())
    for oid in orders:
        od = "".join(ch for ch in oid if ch.isdigit())
        if digits and od and int(od) == int(digits):
            return oid
    return None


def _flagged(planner: Planner, plan: dict, scenario: dict | None) -> list[dict]:
    """Late or unserved orders, each with its closest option (ignoring other orders' capacity)."""
    pdata = planner.problem(scenario)
    out = []
    for a in plan["allocation"]:
        if not a["on_time"]:
            opts = option_table(pdata, a["order_id"])
            on_time_somewhere = any(o["can_be_on_time"] for o in opts)
            out.append({"order_id": a["order_id"], "issue": "late", "plant": a["plant"], "arrival": a["arrival"],
                        "due": a["due"], "late_days": a["late_days"],
                        "cause": "capacity is full in the plants that could deliver on time" if on_time_somewhere
                        else "no plant can deliver by the due date",
                        "closest_option": min(opts, key=lambda o: (o["late_days"], o["cost"])) if opts else None})
    for u in plan["unserved"]:
        opts = option_table(pdata, u["order_id"])
        out.append({"order_id": u["order_id"], "issue": "unserved", "due": u["due"], "cause": u["reason"],
                    "closest_option": min(opts, key=lambda o: (o["late_days"], o["cost"])) if opts else None})
    return out


# --------------------------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------------------------
def intake(state: State) -> State:
    ctx = state["ctx"]
    t = time.perf_counter()
    parsed, info = intent_mod.parse(ctx.request)
    p = parsed.model_dump()
    ctx.llm_calls += info["llm_calls"]
    ctx.tokens += info["tokens"]
    # Structured inputs in the request body win over anything read from the text.
    w_in = {k: v for k, v in (state.get("weights_in") or {}).items() if v is not None}
    if w_in:
        p["weights_cost"], p["weights_carbon"] = w_in.get("cost", p["weights_cost"]), w_in.get("carbon", p["weights_carbon"])
        p["weights_lateness"] = w_in.get("lateness")
    sc = state.get("scenario_in") or {}
    if sc:
        from src.cost_model import normalise_scenario
        ns = normalise_scenario(sc)
        if ns["capacity_changes"]:
            p["capacity_changes"] = [{"plant": c["plant"], "loss_pct": c["loss"] * 100, "weeks": c.get("weeks")}
                                     for c in ns["capacity_changes"]]
            if p["intent"] in ("unknown", "allocation"):
                p["intent"] = "replan"
        if ns["carbon_cut_pct"] is not None:
            p["carbon_cut_pct"] = float(ns["carbon_cut_pct"])
            if p["intent"] in ("unknown", "allocation"):
                p["intent"] = "tradeoff"
        if isinstance(ns["force"], dict) and ns["force"].get("order_id"):
            p["order_id"], p["force_plant"] = ns["force"]["order_id"], str(ns["force"].get("plant", "")).upper()
            p["intent"] = "scenario"
    ctx.record("intake", "parse_intent", ctx.request, p, round((time.perf_counter() - t) * 1000, 1),
               method=info["method"], llm_calls=info["llm_calls"], tokens=info["tokens"],
               **({"errors": info["errors"]} if info.get("errors") else {}))
    return {"parsed": p, "intent": p["intent"]}


CAPABILITIES = ("allocate orders, find the cheapest on-time plan, price a carbon cut, re-plan after a plant "
                "loses capacity, explain why an order went to a plant, or write the management brief")


def guard(state: State) -> State:
    """Deterministic rules on the *request* (never on a plan)."""
    ctx, p = state["ctx"], state["parsed"]
    t = time.perf_counter()
    decision, msg = "proceed", ""
    intent = p["intent"]
    try:
        ctx.planner = Planner(ctx.conn)
        pdata = ctx.planner.problem()
    except Exception as exc:
        pdata = None
        decision, msg = "clarification_needed", f"No usable data is loaded yet ({exc}). Load the CSV files through POST /ingest or the loader first."
    if pdata is not None:
        plants = set(pdata.plants)
        oid = _find_order(pdata.orders, p.get("order_id"))
        weights = {"cost": p.get("weights_cost"), "carbon": p.get("weights_carbon"), "lateness": p.get("weights_lateness")}
        if p.get("asks_to_break_rule"):
            decision, msg = "refused", ("This request asks to break an allocation rule (capacity, eligibility or due dates). "
                                        "Plans must respect every rule, so it cannot be done. You can ask for a re-plan "
                                        "with changed capacity, or for the cost of other options, instead.")
        elif intent == "unknown":
            decision, msg = "clarification_needed", f"I could not tell what you need. I can {CAPABILITIES}."
        elif p.get("asks_weight_change_without_values"):
            decision, msg = "clarification_needed", ("Which weights should I use? Give cost and carbon weights, "
                                                     "for example 60/40 cost/carbon.")
        elif (weights["cost"] or 0) == 0 and (weights["carbon"] or 0) == 0 and weights["cost"] is not None and weights["carbon"] is not None:
            decision, msg = "clarification_needed", "Cost and carbon weights cannot both be zero. Which matters more?"
        elif intent in ("explain", "scenario") and p.get("order_id") and not oid:
            decision, msg = "clarification_needed", f"Order {p['order_id']} is not in the loaded data. Which order do you mean?"
        elif intent == "explain" and not oid:
            decision, msg = "clarification_needed", "Which order should I explain? Give its order id, for example O-077."
        elif p.get("force_plant"):
            fp = p["force_plant"]
            product = pdata.orders[oid]["product"] if oid else None
            if fp not in plants:
                decision, msg = "clarification_needed", f"Plant {fp} is not in the loaded data (plants: {sorted(plants)})."
            elif oid and not any(o.plant == fp for o in pdata.options[oid]):
                decision, msg = "refused", (f"Plant {fp} cannot build order {oid}: it cannot make product {product} "
                                            f"or has no route to {pdata.orders[oid]['destination']}. Assigning it there "
                                            f"would break the eligibility rule.")
        if decision == "proceed":
            for c in p.get("capacity_changes") or []:
                if c["plant"] not in plants:
                    decision, msg = "clarification_needed", f"Plant {c['plant']} is not in the loaded data (plants: {sorted(plants)}). Which plant?"
                elif not 0 < float(c["loss_pct"]) <= 100:
                    decision, msg = "clarification_needed", f"A capacity loss of {c['loss_pct']}% is not possible. What share of capacity is lost (0-100%)?"
        if decision == "proceed" and intent == "replan" and not p.get("capacity_changes") and not pdata.events_applied:
            decision, msg = "clarification_needed", ("Which plant loses capacity, by how much, and for which weeks? "
                                                     "No capacity event is loaded yet.")
        if decision == "proceed" and p.get("order_id") and oid:
            p["order_id"] = oid
    ctx.record("guard", "check_request", {"intent": intent}, {"decision": decision, "message": msg},
               round((time.perf_counter() - t) * 1000, 1))
    return {"decision": decision, "message": msg, "parsed": p}


def route_after_guard(state: State) -> str:
    if state["decision"] != "proceed":
        return "writer"
    i = state["intent"]
    if i in ("allocation", "cheapest_on_time") or (i == "scenario" and state["parsed"].get("force_plant")):
        return "optimiser"
    if i in ("replan", "tradeoff", "scenario"):
        return "scenario_agent"
    return {"explain": "explain", "brief": "brief"}.get(i, "optimiser")


def _weights(p: dict) -> dict:
    return resolve_weights({"cost": p.get("weights_cost"), "carbon": p.get("weights_carbon"),
                            "lateness": p.get("weights_lateness")})


def optimiser(state: State) -> State:
    ctx, p = state["ctx"], state["parsed"]
    w = _weights(p)
    on_time = p["intent"] == "cheapest_on_time"
    if on_time:
        w = resolve_weights({"cost": 1.0, "carbon": 0.0, "lateness": w["lateness"]})
    label = "cheapest on-time plan" if on_time else "plan"
    plan = _timed(ctx, "optimiser", "solve_allocation",
                  lambda: ctx.planner.plan(None, w, on_time_only=on_time, label=label),
                  {"weights": w, "on_time_only": on_time})
    out: State = {"weights": w, "primary_plan": plan, "base_plan": plan}
    if p.get("force_plant"):
        oid, fp = p["order_id"], p["force_plant"]
        forced = _timed(ctx, "optimiser", "solve_allocation",
                        lambda: ctx.planner.plan(None, w, force={oid: fp}, label=f"{oid} forced to plant {fp}"),
                        {"weights": w, "force": {oid: fp}})
        comp = _timed(ctx, "scenario", "compare_plans", lambda: compare_plans(plan, forced), {"base": "plan", "new": "forced"})
        out.update(primary_plan=forced, comparison=comp)
    return out


def scenario_agent(state: State) -> State:
    ctx, p = state["ctx"], state["parsed"]
    w = _weights(p)
    replan_events = p["intent"] == "replan" and not p.get("capacity_changes")
    base_scenario = {"ignore_events": True} if replan_events else {}
    base = _timed(ctx, "optimiser", "solve_allocation",
                  lambda: ctx.planner.plan(base_scenario, w, label="base plan" + (" (before capacity events)" if replan_events else "")),
                  {"weights": w, "scenario": base_scenario})
    runner = ScenarioRunner(ctx.planner, base_scenario, w, ctx.record)
    runner.add_base(base)
    info = run_agent(runner, ctx.request, p["intent"], p)
    ctx.llm_calls += info["llm_calls"]
    ctx.tokens += info["tokens"]
    ctx.record("scenario", "agent_summary", None, info)
    primary = runner.plans.get(runner.last_plan_id) if runner.last_plan_id else base
    tradeoff = runner.frontier["points"] if runner.frontier else []
    return {"weights": w, "base_plan": base, "primary_plan": primary,
            "comparison": runner.comparisons[-1] if runner.comparisons else None, "tradeoff": tradeoff,
            "_scenario_for_primary": runner.scenarios.get(runner.last_plan_id or "base")}


def explain(state: State) -> State:
    ctx, p = state["ctx"], state["parsed"]
    w = _weights(p)
    oid = p["order_id"]
    plan = _timed(ctx, "optimiser", "solve_allocation", lambda: ctx.planner.plan(None, w, label="plan"), {"weights": w})
    pdata = ctx.planner.problem()
    o = pdata.orders[oid]
    table = _timed(ctx, "explainer", "option_table", lambda: option_table(pdata, oid), {"order_id": oid})
    row = next((a for a in plan["allocation"] if a["order_id"] == oid), None)
    util = {(u["plant"], u["week"]): u for u in plan["weekly_utilisation"]}
    alternatives = []
    makers = {t["plant"] for t in table}
    for plant in pdata.plants:
        if row and plant == row["plant"]:
            continue
        if plant not in makers:
            alternatives.append({"plant": plant, "eligible": False,
                                 "reason": f"cannot make {o['product']} or has no route to {o['destination']}."})
            continue
        t = next(t for t in table if t["plant"] == plant)
        forced = _timed(ctx, "explainer", "solve_allocation",
                        lambda: ctx.planner.plan(None, w, force={oid: plant}, label=f"{oid} at {plant}"),
                        {"force": {oid: plant}})
        d_cost = round(forced["kpis"]["total_cost"] - plan["kpis"]["total_cost"], 2)
        d_carbon = round(forced["kpis"]["total_carbon_t"] - plan["kpis"]["total_carbon_t"], 3)
        d_late = forced["kpis"]["late_orders"] - plan["kpis"]["late_orders"]
        parts = []
        extra: dict[str, Any] = {}
        if row:
            dc = round(t["cost"] - row["cost"], 2)
            extra["order_cost_delta"] = dc
            parts.append(f"order cost {writer.money(t['cost'])} ({'+' if dc >= 0 else ''}{writer.money(dc)} vs chosen)")
            parts.append(f"carbon {t['carbon_kg']:,.1f} kg")
        parts.append(f"best arrival {t['arrival']}" + (f", {t['late_days']} days late" if t["late_days"] else " (on time)"))
        wk = util.get((plant, t["week"]))
        if wk and wk["capacity"] and wk["used"] >= 0.98 * wk["capacity"]:
            extra.update(week_used=wk["used"], week_capacity=wk["capacity"])
            parts.append(f"week {t['week']} capacity is full ({wk['used']:.0f}/{wk['capacity']:.0f} units)")
        parts.append(f"moving it there changes the whole plan by {writer.money(d_cost)} cost, {d_carbon:+.3f} t carbon, "
                     f"{d_late:+d} late orders")
        alternatives.append({"plant": plant, "eligible": True, **t, **extra, "plan_cost_delta": d_cost,
                             "plan_carbon_delta_t": d_carbon, "plan_late_delta": d_late, "reason": "; ".join(parts) + "."})
    chosen = row or {"plant": None, "week": None, "cost": 0, "carbon_kg": 0, "arrival": None, "on_time": False}
    explanation = {"order": {"order_id": oid, "product": o["product"], "qty": o["qty"],
                             "destination": o["destination"], "due": o["due"].isoformat()},
                   "chosen": chosen, "options": table, "alternatives": alternatives}
    return {"weights": w, "primary_plan": plan, "base_plan": plan, "explanation": explanation}


def brief(state: State) -> State:
    ctx, p = state["ctx"], state["parsed"]
    w = _weights(p)
    plan = _timed(ctx, "optimiser", "solve_allocation", lambda: ctx.planner.plan(None, w, label="recommended plan"), {"weights": w})
    front = _timed(ctx, "scenario", "tradeoff_frontier", lambda: ctx.planner.frontier(None, [5, 10, 15, 20, 25, 30], w),
                   {"cuts": [5, 10, 15, 20, 25, 30]})
    comp = None
    if ctx.planner.problem().events_applied:
        before = _timed(ctx, "optimiser", "solve_allocation",
                        lambda: ctx.planner.plan({"ignore_events": True}, w, label="before capacity events"),
                        {"scenario": {"ignore_events": True}})
        comp = _timed(ctx, "scenario", "compare_plans", lambda: compare_plans(before, plan), {"base": "before events", "new": "recommended"})
    return {"weights": w, "primary_plan": plan, "base_plan": plan, "tradeoff": front["points"], "comparison": comp}


def checker(state: State) -> State:
    ctx = state["ctx"]
    plan = state.get("primary_plan")
    if not plan:
        return {}
    scenario = state.get("_scenario_for_primary")
    pdata = ctx.planner.problem(scenario)
    res = _timed(ctx, "checker", "validate_plan", lambda: validate_plan(ctx.planner.frames(), pdata, plan),
                 {"plan": plan.get("label")})
    return {"validator": res}


def _answer_type(state: State) -> str:
    if state["decision"] != "proceed":
        return state["decision"]
    i = state["intent"]
    plan = state.get("primary_plan") or {}
    unserved = (plan.get("kpis") or {}).get("unserved_orders", 0)
    if i == "brief":
        return "brief"
    if i in ("replan", "tradeoff", "scenario"):
        base_unserved = (state.get("base_plan") or {}).get("kpis", {}).get("unserved_orders", 0)
        return "infeasible" if unserved > base_unserved else "scenario"
    return "infeasible" if unserved else "allocation"


def build_facts(state: State) -> State:
    ctx = state["ctx"]
    at = _answer_type(state)
    facts: dict[str, Any] = {"answer_type": at}
    if state["decision"] != "proceed":
        facts["message"] = state["message"]
        return {"facts": facts, "answer_type": at}
    plan = state["primary_plan"]
    i = state["intent"]
    flagged = _flagged(ctx.planner, plan, state.get("_scenario_for_primary"))
    k = dict(plan["kpis"])
    facts.update(kpis=k, weights_used=plan["weights"], n_plants=len(k["plant_utilisation"]),
                 flagged_count=len(flagged), flagged_orders=flagged[:10], solver_status=plan["status"],
                 solver_gap=plan.get("gap"))
    if i == "explain":
        facts.update(kind="explain", explanation=state["explanation"])
    elif i == "brief":
        facts.update(kind="brief", tradeoff=state.get("tradeoff", []))
        c = state.get("comparison")
        if c:
            facts["capacity_event_impact"] = {kk: v for kk, v in c.items() if kk != "moved_orders"}
    elif i == "tradeoff" and state.get("tradeoff"):
        facts.update(kind="tradeoff", tradeoff=state["tradeoff"], requested_cut_pct=state["parsed"].get("carbon_cut_pct"))
        if state.get("comparison"):
            facts["comparison"] = {kk: v for kk, v in state["comparison"].items() if kk != "moved_orders"}
    elif state.get("comparison"):
        c = state["comparison"]
        label = plan.get("label", "Scenario")
        changes = state["parsed"].get("capacity_changes") or []
        if changes:
            label = "; ".join(f"Plant {c_['plant']} loses {c_['loss_pct']:g}%" for c_ in changes)
        facts.update(kind="scenario", scenario_label=label, capacity_changes=changes,
                     comparison={**{kk: v for kk, v in c.items() if kk != "moved_orders"},
                                 "moved_orders": c["moved_orders"][:10]})
    else:
        facts.update(kind="allocation",
                     plan_title="Cheapest on-time plan" if i == "cheapest_on_time" else "Plan")
    return {"facts": facts, "answer_type": at}


def writer_node(state: State) -> State:
    ctx = state["ctx"]
    t = time.perf_counter()
    text, info = writer.write(state["answer_type"], state["facts"], ctx.request, state.get("feedback"))
    ctx.llm_calls += info["llm_calls"]
    ctx.tokens += info["tokens"]
    ctx.record("writer", "write_summary" if state["facts"].get("kind") != "brief" else "write_brief",
               {"answer_type": state["answer_type"]}, {"text": text}, round((time.perf_counter() - t) * 1000, 1),
               method=info["method"], llm_calls=info["llm_calls"], tokens=info["tokens"])
    return {"text": text, "write_attempts": state.get("write_attempts", 0) + 1}


def number_check(state: State) -> State:
    ctx = state["ctx"]
    if state["answer_type"] in ("clarification_needed", "refused"):
        res = {"ok": True, "numbers_checked": 0, "unverified": [], "faithfulness": 1.0, "note": "fixed message, not generated"}
        ctx.record("checker", "number_check", None, res)
        return {"number_check": res}
    res = check_numbers(state["text"], state["facts"])
    out: State = {"number_check": res}
    if not res["ok"] and state.get("write_attempts", 0) >= 2:
        text = writer.template(state["answer_type"], state["facts"])
        res2 = check_numbers(text, state["facts"])
        out = {"text": text, "number_check": {**res2, "fallback_used": True, "rejected_numbers": res["unverified"]}}
    elif not res["ok"]:
        out["feedback"] = res["unverified"]
    ctx.record("checker", "number_check", None, out["number_check"])
    return out


def after_number_check(state: State) -> str:
    nc = state["number_check"]
    return "respond" if nc["ok"] or nc.get("fallback_used") else "writer"


def _recommendations(state: State, run_id: str) -> list[dict]:
    if state["decision"] != "proceed":
        return []
    plan, v = state["primary_plan"], state.get("validator") or {}
    k = plan["kpis"]
    ok = v.get("valid", False) and plan["status"] in ("OPTIMAL", "FEASIBLE")
    conf = "high" if ok and k["late_orders"] == 0 and not k["unserved_orders"] else ("medium" if ok and not k["unserved_orders"] else "low")
    recs = []
    c = state.get("comparison")
    if c and state["intent"] in ("replan", "scenario", "tradeoff"):
        recs.append({"action": f"Adopt the '{plan['label']}' plan: move {c['orders_moved']} orders as listed in moved_orders.",
                     "rationale": f"Cost changes by {c['cost_delta']:,.2f} ({c['cost_delta_pct']:+.2f}%), carbon by "
                                  f"{c['carbon_delta_t']:+.3f} t, late orders by {c['late_orders_delta']:+d}.",
                     "evidence_refs": [f"run:{run_id}", "comparison", "kpis", "validator"], "confidence": conf,
                     "estimated_impact": {"cost_delta": c["cost_delta"], "carbon_delta_t": c["carbon_delta_t"],
                                          "late_orders_delta": float(c["late_orders_delta"])},
                     "requires_approval": c["cost_delta"] > 0 or c["orders_moved"] > 0})
    elif state["intent"] in ("allocation", "cheapest_on_time", "brief", "explain", "scenario"):
        recs.append({"action": "Release the plan: " + ", ".join(f"Plant {p} {n} orders" for p, n in k["orders_by_plant"].items()) + ".",
                     "rationale": f"Lowest weighted cost/carbon plan found by the solver ({plan['status']}, gap {plan.get('gap')}) "
                                  f"that passes the capacity and eligibility checks.",
                     "evidence_refs": [f"run:{run_id}", "kpis", "validator"], "confidence": conf,
                     "estimated_impact": {"total_cost": k["total_cost"], "total_carbon_t": k["total_carbon_t"],
                                          "late_orders": float(k["late_orders"])},
                     "requires_approval": bool(k["late_orders"] or k["unserved_orders"])})
    flagged = state["facts"].get("flagged_count", 0)
    if flagged:
        recs.append({"action": f"Agree new dates or expediting with customers for {flagged} late or unserved orders.",
                     "rationale": "These orders cannot be delivered on time within capacity; flagged_orders lists the closest option.",
                     "evidence_refs": [f"run:{run_id}", "flagged_orders"], "confidence": "high",
                     "estimated_impact": {"orders_at_risk": float(flagged)}, "requires_approval": True})
    return recs


def respond(state: State) -> State:
    ctx = state["ctx"]
    at = state["answer_type"]
    plan = state.get("primary_plan")
    assumptions, limitations = [], []
    if ctx.planner is not None:
        try:
            pd_ = ctx.planner.problem(state.get("_scenario_for_primary"))
            assumptions, limitations = list(pd_.assumptions), list(pd_.limitations)
        except Exception:
            pass
    w = state.get("weights") or {}
    if w:
        assumptions.insert(0, f"Weights {w.get('cost', 0) * 100:.0f}% cost / {w.get('carbon', 0) * 100:.0f}% carbon; "
                              f"lateness weight {w.get('lateness')} (a late order costs as much as ~1 average order's normalised cost).")
    if plan and plan.get("status") == "FEASIBLE":
        limitations.append(f"Solver stopped at its fixed work budget; the plan is within a proven "
                           f"{(plan.get('gap') or 0) * 100:.2f}% of the best possible objective.")
    limitations.append("Carbon covers plant energy and transport only, using the factors in the data.")
    v = state.get("validator")
    resp: dict[str, Any] = {
        "run_id": ctx.run_id, "answer_type": at, "summary": state.get("text", ""),
        "allocation": plan["allocation"] if plan else [],
        "kpis": plan["kpis"] if plan else None,
        "tradeoff": state.get("tradeoff") or [],
        "weights_used": w,
        "validator": ({"capacity_ok": v["capacity_ok"], "eligibility_ok": v["eligibility_ok"],
                       "delivery_ok": v["delivery_ok"], "cost_ok": v["cost_ok"], "valid": v["valid"],
                       "violations": v["violations"]} if v else None),
        "recommendations": _recommendations(state, ctx.run_id),
        "flagged_orders": state["facts"].get("flagged_orders", []) if state["decision"] == "proceed" else [],
        "assumptions": assumptions, "limitations": limitations,
        "number_check": state.get("number_check"),
    }
    if state.get("comparison"):
        resp["comparison"] = state["comparison"]
        resp["moved_orders"] = state["comparison"]["moved_orders"]
    if state.get("explanation"):
        resp["explanation"] = state["explanation"]
    if plan:
        resp["unserved"] = plan["unserved"]
        resp["solver"] = {"status": plan["status"], "gap": plan.get("gap"), "solve_ms": plan.get("solve_ms"),
                          "backend": plan.get("solver")}
    latency = round((time.perf_counter() - ctx.t0) * 1000, 1)
    resp["latency_ms"] = latency
    trace = {"run_id": ctx.run_id, "request": ctx.request, "created_at": datetime.now(timezone.utc).isoformat(),
             "answer_type": at, "steps": ctx.steps, "llm_calls": ctx.llm_calls, "tokens": ctx.tokens,
             "latency_ms": latency, "data_version": get_data_version(ctx.conn),
             "config": {"llm_provider": settings.LLM_PROVIDER, "llm_model": settings.LLM_MODEL,
                        "mock_llm": settings.MOCK_LLM, "solver": settings.SOLVER,
                        "solver_time_limit_s": settings.SOLVER_TIME_LIMIT_S}}
    resp["trace_summary"] = {"llm_calls": ctx.llm_calls, "tokens": ctx.tokens, "steps": len(ctx.steps)}
    save_run(ctx.conn, ctx.run_id, ctx.request, at, resp, trace)
    return {"response": resp}


# --------------------------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------------------------
def build_graph():
    g = StateGraph(State)
    for name, fn in [("intake", intake), ("guard", guard), ("optimiser", optimiser), ("scenario_agent", scenario_agent),
                     ("explain", explain), ("brief", brief), ("checker", checker), ("facts", build_facts),
                     ("writer", writer_node), ("number_check", number_check), ("respond", respond)]:
        g.add_node(name, fn)
    g.set_entry_point("intake")
    g.add_edge("intake", "guard")
    g.add_conditional_edges("guard", route_after_guard,
                            {"optimiser": "optimiser", "scenario_agent": "scenario_agent", "explain": "explain",
                             "brief": "brief", "writer": "facts"})
    for n in ("optimiser", "scenario_agent", "explain", "brief"):
        g.add_edge(n, "checker")
    g.add_edge("checker", "facts")
    g.add_edge("facts", "writer")
    g.add_edge("writer", "number_check")
    g.add_conditional_edges("number_check", after_number_check, {"respond": "respond", "writer": "writer"})
    g.add_edge("respond", END)
    return g.compile()


GRAPH = build_graph()


def run(request: str, weights: dict | None = None, scenario: dict | None = None, conn=None) -> dict:
    conn = conn or get_connection()
    ctx = RunContext(run_id=f"al-{uuid.uuid4().hex[:6]}", request=request, conn=conn)
    out = GRAPH.invoke({"ctx": ctx, "weights_in": weights, "scenario_in": scenario, "write_attempts": 0})
    return out["response"]
