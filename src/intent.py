"""Step 1: turn the request text into structured parameters.

The LLM fills a small Pydantic schema (no numbers are computed here). If MOCK_LLM is on, or the LLM
fails twice, a rule-based parser fills the same schema, so the pipeline never depends on the LLM.
"""
from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field

from src import llm

IntentName = Literal["allocation", "cheapest_on_time", "tradeoff", "replan", "scenario",
                     "explain", "brief", "unknown"]


class CapacityChange(BaseModel):
    plant: str = Field(description="plant letter, e.g. 'B'")
    loss_pct: float = Field(description="share of capacity lost, in percent (30 means 30%)")
    weeks: Optional[list[int]] = Field(default=None, description="1-based planning weeks affected, if stated")


class ParsedRequest(BaseModel):
    intent: IntentName = Field(description=(
        "allocation: make/show a plan; cheapest_on_time: cheapest plan with every order on time; "
        "tradeoff: cost of cutting carbon or the cost/carbon trade-off; replan: a plant loses capacity "
        "or re-plan after a capacity event; scenario: other what-if (weights change, force an order to a plant); "
        "explain: why an order went to a plant; brief: management/leadership brief; unknown: none of these"))
    order_id: Optional[str] = Field(default=None, description="order id mentioned, e.g. 'O-077' or '77'")
    weights_cost: Optional[float] = Field(default=None, description="cost weight if the user gave one (0-1)")
    weights_carbon: Optional[float] = Field(default=None, description="carbon weight if the user gave one (0-1)")
    capacity_changes: list[CapacityChange] = Field(default_factory=list)
    carbon_cut_pct: Optional[float] = Field(default=None, description="requested carbon cut in percent")
    force_plant: Optional[str] = Field(default=None, description="plant the user wants the order forced to")
    asks_to_break_rule: bool = Field(default=False, description=(
        "true only if the user asks to exceed capacity, ignore eligibility or due dates, or invent numbers"))
    asks_weight_change_without_values: bool = Field(default=False, description=(
        "true if the user wants different weights but gives no values"))
    claimed_numbers: list[str] = Field(default_factory=list, description=(
        "numbers the user asserts as facts and wants repeated (e.g. 'say we saved 5M')"))


SYSTEM = """You convert a planning request for a 3-plant manufacturer into JSON parameters.
Never compute or guess numbers. Only copy numbers the user wrote. Leave fields empty when not stated."""

RULE_BREAK = re.compile(r"ignore (the )?(capacity|due|deadline|eligib|rules?)|exceed|beyond (its |the )?capacity|"
                        r"over (its |the )?capacity|more than (its|the) capacity|skip (the )?(validat|check)|"
                        r"(make|invent|fabricate) up? ?(the )?numbers|regardless of capacity|even if .*can'?t make")


def _pct(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*(?:%|percent|pct)", text)]


def rule_parse(request: str) -> ParsedRequest:
    """Deterministic parser: used in MOCK_LLM mode and as the fallback when the LLM fails."""
    t = request.lower().strip()
    p = ParsedRequest(intent="unknown")

    m = re.search(r"\border\b\s*(?:id\s*)?#?\s*([a-z0-9][a-z0-9_-]*)", t)
    if m and m.group(1) not in {"in", "to", "the", "was", "is", "at", "be", "from", "for", "built", "made"}:
        p.order_id = m.group(1).upper()
    elif re.search(r"\b(o-\d+)\b", t):
        p.order_id = re.search(r"\b(o-\d+)\b", t).group(1).upper()
    plants = re.findall(r"\bplant\s+([a-z])\b", t)
    pcts = _pct(t)

    # weights: "70/30", "cost 0.6 carbon 0.4", "60% cost", "carbon weight 0.5"
    m = re.search(r"\b(\d{1,3})\s*/\s*(\d{1,3})\b", t)
    if m and ("weight" in t or "cost" in t):
        a, b = float(m.group(1)), float(m.group(2))
        first_cost = t.find("cost") <= t.find("carbon") if "carbon" in t and "cost" in t else True
        c, e = (a, b) if first_cost else (b, a)
        p.weights_cost, p.weights_carbon = c / (a + b), e / (a + b)
    for name in ("cost", "carbon") if ("weight" in t or re.search(r"cost\s*[=:]?\s*0?\.\d", t)) else ():
        m = re.search(rf"{name}\s*(?:weight)?\s*(?:=|:|of|at|to)?\s*(0?\.\d+|1\.0|\d{{1,3}}\s*%)", t) or \
            re.search(rf"(0?\.\d+|\d{{1,3}}\s*%)\s*(?:weight\s*(?:on|for)\s*)?{name}", t)
        if m:
            v = m.group(1).replace(" ", "")
            val = float(v[:-1]) / 100 if v.endswith("%") else float(v)
            setattr(p, f"weights_{name}", val)
    if re.search(r"(change|different|other|new) weights?|re-?weight", t) and p.weights_cost is None and p.weights_carbon is None:
        p.asks_weight_change_without_values = True

    if RULE_BREAK.search(t):
        p.asks_to_break_rule = True
    p.claimed_numbers = re.findall(r"(?:say|claim|state|write|tell them)[^.]*?(\d[\d,\.]*\s*(?:%|m|k|million|t|tonnes)?)", t)

    capacity_words = re.search(r"\b(lose|loses|lost|losing|loss|outage|down|shut|closes|closed|drop|drops|reduc|cut)\w*", t)
    if plants and capacity_words and "carbon" not in t[max(0, capacity_words.start() - 15):capacity_words.end() + 15]:
        loss = pcts[0] if pcts else (100.0 if re.search(r"shut|closed|closes|down completely|offline", t) else None)
        weeks = [int(w) for w in re.findall(r"week\s*(\d+)", t)] or None
        m2 = re.search(r"(?:for|next)\s+(\d+|two|three|one)\s+weeks?", t)
        if not weeks and m2:
            n = {"one": 1, "two": 2, "three": 3}.get(m2.group(1), None) or int(m2.group(1))
            weeks = list(range(1, n + 1))
        if loss is not None:
            p.capacity_changes = [CapacityChange(plant=plants[0].upper(), loss_pct=loss, weeks=weeks)]

    force = re.search(r"\b(put|move|build|assign|make|produce|send)\b.*\border\b.*\b(in|to|at|into|from)\s+plant\s+([a-z])\b", t) \
        or re.search(r"\border\b.*\b(be )?(built|made|produced|moved|assigned)\s+(in|at|to)\s+plant\s+([a-z])\b", t)
    if force and p.order_id and not t.startswith("why"):
        p.force_plant = force.groups()[-1].upper()

    if re.search(r"\bwhy\b", t) and (p.order_id or "order" in t):
        p.intent = "explain"
    elif re.search(r"brief|leadership|management|executive|board|summary for", t):
        p.intent = "brief"
    elif p.capacity_changes or re.search(r"re-?plan|capacity (event|loss|drop)|loses capacity|data drop", t):
        p.intent = "replan"
    elif re.search(r"carbon\s*(cut|reduc)|cut\s*(the\s*)?carbon|reduce\s*(the\s*)?carbon|lower\s*carbon|greener|"
                   r"trade-?off|frontier|pareto|cost of (cutting|reducing)", t):
        p.intent = "tradeoff"
        cut = [x for x in pcts if 0 < x < 100]
        p.carbon_cut_pct = cut[0] if cut else None
    elif p.force_plant:
        p.intent = "scenario"
    elif "cheapest" in t and re.search(r"on[- ]time|deliver|due|date", t):
        p.intent = "cheapest_on_time"
    elif re.search(r"what if|what-if|scenario|compare", t) and (p.weights_cost is not None or p.weights_carbon is not None):
        p.intent = "scenario"
    elif re.search(r"allocat|assign|plan|schedule|orders|cheapest|weights?", t):
        p.intent = "allocation"
        if "cheapest" in t and p.weights_cost is None:
            p.weights_cost, p.weights_carbon = 1.0, 0.0
    return p


def parse(request: str) -> tuple[ParsedRequest, dict]:
    """Returns (parsed request, trace info). Uses the LLM when enabled, with retry and rule fallback."""
    info = {"method": "rules", "llm_calls": 0, "tokens": 0, "errors": []}
    if not llm.llm_enabled():
        return rule_parse(request), info
    for attempt in range(2):
        try:
            parsed, tok = llm.structured(SYSTEM, request, ParsedRequest)
            info.update(method="llm", llm_calls=info["llm_calls"] + 1, tokens=info["tokens"] + tok)
            return parsed, info
        except Exception as exc:  # bad JSON, timeout, missing key...
            info["llm_calls"] += 1
            info["errors"].append(f"attempt {attempt + 1}: {str(exc)[:200]}")
    info["method"] = "rules (LLM failed twice)"
    return rule_parse(request), info
