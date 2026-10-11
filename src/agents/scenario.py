"""Scenario agent (R9). The LLM decides which solver runs a what-if needs, by calling tools.

Tools (all numbers come from these; the LLM cannot write a plan):
    solve_plan(label, capacity_changes, carbon_cut_pct, cost_weight, carbon_weight, on_time_only)
    compare_plans(base_id, new_id)
    tradeoff_frontier(cuts)

At most SCENARIO_MAX_TOOL_CALLS calls. In MOCK_LLM mode, or if the LLM misbehaves, a fixed tool
sequence for the parsed intent is run instead, through the same tool functions and the same trace.
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from src import llm
from src.config import settings
from src.planning import Planner
from src.solver import compare_plans as _compare
from src.solver import plan_assignment


class CapChange(BaseModel):
    plant: str
    loss_pct: float = Field(description="percent of capacity lost, 30 = 30%")
    weeks: Optional[list[int]] = Field(default=None, description="1-based weeks; omit for the whole horizon")


class solve_plan(BaseModel):
    """Run the allocation solver on a changed scenario. Returns the plan id and its KPIs."""
    label: str = Field(description="short name for this plan")
    capacity_changes: list[CapChange] = Field(default_factory=list)
    carbon_cut_pct: Optional[float] = Field(default=None, description="cut carbon by this percent vs the cheapest plan")
    cost_weight: Optional[float] = None
    carbon_weight: Optional[float] = None
    on_time_only: bool = False
    apply_capacity_events: bool = Field(default=False, description=(
        "true to apply the capacity events loaded from capacity_events.csv (the base plan ignores them)"))


class compare_plans(BaseModel):
    """Compare two plans by id. Returns cost, carbon and lateness deltas and the orders that moved."""
    base_id: str
    new_id: str


class tradeoff_frontier(BaseModel):
    """Cost of cutting carbon by each percent in `cuts` (epsilon-constraint on the solver)."""
    cuts: list[float] = Field(default_factory=lambda: [5, 10, 15, 20, 25, 30])


SYSTEM = """You run what-if scenarios for a 3-plant production plan by calling tools.
A plan with id 'base' already exists. If the base ignores the loaded capacity events and the user asks to
re-plan for them, call solve_plan with apply_capacity_events=true. Call solve_plan for the changed scenario, then compare_plans('base', <new id>).
For questions about the cost of cutting carbon, call tradeoff_frontier (and solve_plan with carbon_cut_pct if one
specific cut was asked). Never invent numbers. Stop calling tools when you have what the question needs,
and answer with one short sentence naming the plan ids you produced."""


class ScenarioRunner:
    """Holds the plans produced during one request and executes tool calls."""

    def __init__(self, planner: Planner, base_scenario: dict, base_weights: dict, record: Callable[..., None]):
        self.planner = planner
        self.base_scenario = base_scenario
        self.base_weights = base_weights
        self.record = record
        self.plans: dict[str, dict] = {}
        self.scenarios: dict[str, dict] = {}
        self.comparisons: list[dict] = []
        self.frontier: Optional[dict] = None
        self.last_plan_id: Optional[str] = None

    def add_base(self, plan: dict) -> None:
        self.plans["base"] = plan
        self.scenarios["base"] = self.base_scenario

    def run(self, name: str, args: dict) -> dict:
        t0 = time.perf_counter()
        if name == "solve_plan":
            out = self._solve(solve_plan(**args))
        elif name == "compare_plans":
            out = self._compare(compare_plans(**args))
        elif name == "tradeoff_frontier":
            out = self._frontier(tradeoff_frontier(**args))
        else:
            out = {"error": f"unknown tool {name}"}
        self.record(agent="scenario", tool=name, input=args, output=out,
                    duration_ms=round((time.perf_counter() - t0) * 1000, 1))
        return out

    def _solve(self, a: solve_plan) -> dict:
        changes = list(self.base_scenario.get("capacity_changes", []))
        changes += [{"plant": c.plant.upper().replace("PLANT", "").strip(), "loss_pct": c.loss_pct, "weeks": c.weeks}
                    for c in a.capacity_changes]
        scenario = dict(self.base_scenario, capacity_changes=changes)
        if a.apply_capacity_events:
            scenario["ignore_events"] = False
        weights = dict(self.base_weights)
        if a.cost_weight is not None:
            weights["cost"] = a.cost_weight
        if a.carbon_weight is not None:
            weights["carbon"] = a.carbon_weight
        cap = None
        if a.carbon_cut_pct is not None:
            front = self.planner.frontier(scenario, [a.carbon_cut_pct], weights)
            base_carbon_kg = front["base"]["kpis"]["total_carbon_t"] * 1000
            cap = base_carbon_kg * (1 - a.carbon_cut_pct / 100)
            weights = {"cost": 1.0, "carbon": 0.0, "lateness": weights.get("lateness")}
        baseline = plan_assignment(self.plans["base"]) if (a.capacity_changes or a.apply_capacity_events) else None
        plan = self.planner.plan(scenario, weights, on_time_only=a.on_time_only, carbon_cap_kg=cap,
                                 baseline=baseline, label=a.label)
        pid = f"p{len(self.plans)}"
        self.plans[pid], self.scenarios[pid], self.last_plan_id = plan, scenario, pid
        return {"plan_id": pid, "label": a.label, "status": plan["status"], "kpis": plan["kpis"]}

    def _compare(self, a: compare_plans) -> dict:
        if a.base_id not in self.plans or a.new_id not in self.plans:
            return {"error": f"unknown plan id; have {list(self.plans)}"}
        c = _compare(self.plans[a.base_id], self.plans[a.new_id])
        self.comparisons.append(c)
        return {k: v for k, v in c.items() if k != "moved_orders"} | {"moved_orders_first_10": c["moved_orders"][:10]}

    def _frontier(self, a: tradeoff_frontier) -> dict:
        self.frontier = self.planner.frontier(self.base_scenario, a.cuts, self.base_weights)
        return {"points": self.frontier["points"]}


def deterministic_calls(intent: str, parsed: dict) -> list[tuple[str, dict]]:
    """The fixed tool sequence used without an LLM (and as the fallback)."""
    if intent == "replan":
        changes = parsed.get("capacity_changes", [])
        args = {"label": "after capacity change", "capacity_changes": changes} if changes else \
            {"label": "after capacity events", "apply_capacity_events": True}
        return [("solve_plan", args),
                ("compare_plans", {"base_id": "base", "new_id": "p1"})]
    if intent == "tradeoff":
        cuts = sorted({5, 10, 15, 20, 25, 30} | ({parsed["carbon_cut_pct"]} if parsed.get("carbon_cut_pct") else set()))
        calls: list[tuple[str, dict]] = [("tradeoff_frontier", {"cuts": cuts})]
        if parsed.get("carbon_cut_pct"):
            calls += [("solve_plan", {"label": f"{parsed['carbon_cut_pct']:g}% carbon cut",
                                      "carbon_cut_pct": parsed["carbon_cut_pct"]}),
                      ("compare_plans", {"base_id": "base", "new_id": "p1"})]
        return calls
    # scenario: changed weights
    return [("solve_plan", {"label": "what-if weights", "cost_weight": parsed.get("weights_cost"),
                            "carbon_weight": parsed.get("weights_carbon")}),
            ("compare_plans", {"base_id": "base", "new_id": "p1"})]


def run_agent(runner: ScenarioRunner, request: str, intent: str, parsed: dict) -> dict:
    """Returns info: method, llm_calls, tokens, tool_calls."""
    info: dict[str, Any] = {"method": "fixed tool sequence", "llm_calls": 0, "tokens": 0, "tool_calls": 0}
    if llm.llm_enabled():
        try:
            from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
            model = llm.get_model().bind_tools([solve_plan, compare_plans, tradeoff_frontier])
            base_k = runner.plans["base"]["kpis"]
            msgs: list = [SystemMessage(SYSTEM), HumanMessage(
                f"Question: {request}\nParsed parameters: {json.dumps(parsed)}\n"
                f"Base plan 'base' KPIs: {json.dumps(base_k)}")]
            for _ in range(settings.SCENARIO_MAX_TOOL_CALLS + 1):
                resp = model.invoke(msgs)
                info["llm_calls"] += 1
                info["tokens"] += llm.tokens_of(resp)
                msgs.append(resp)
                if not getattr(resp, "tool_calls", None):
                    break
                for call in resp.tool_calls:
                    if info["tool_calls"] >= settings.SCENARIO_MAX_TOOL_CALLS:
                        out = {"error": "tool call limit reached"}
                    else:
                        out = runner.run(call["name"], call["args"])
                        info["tool_calls"] += 1
                    msgs.append(ToolMessage(json.dumps(out, default=str)[:6000], tool_call_id=call["id"]))
            info["method"] = "llm tool calling"
            if intent != "tradeoff" and runner.last_plan_id is None:
                raise RuntimeError("LLM made no solve_plan call")
            if intent == "tradeoff" and runner.frontier is None and runner.last_plan_id is None:
                raise RuntimeError("LLM made no frontier or solve call")
            if runner.last_plan_id and not runner.comparisons:
                runner.run("compare_plans", {"base_id": "base", "new_id": runner.last_plan_id})
            return info
        except Exception as exc:
            info["error"] = str(exc)[:200]
            info["method"] = "fixed tool sequence (LLM fallback)"
            runner.plans = {"base": runner.plans["base"]}
            runner.comparisons, runner.frontier, runner.last_plan_id = [], None, None
    for name, args in deterministic_calls(intent, parsed):
        runner.run(name, args)
        info["tool_calls"] += 1
    return info
