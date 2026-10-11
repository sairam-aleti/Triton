"""Writer agent (R10). Turns the facts JSON into a summary or a management brief.

The LLM sees only the facts computed by the tools. Its text then goes through the number check;
on failure it gets one retry with the offending numbers listed, then a deterministic template is
used. In MOCK_LLM mode the template is used directly. Templates are built from the same facts, so
they always pass the number check.
"""
from __future__ import annotations

import json
from typing import Any

from src import llm

SYSTEM_SUMMARY = """You write short answers for a manufacturing planning team.
Use ONLY numbers that appear in the FACTS JSON, copied or rounded. Never calculate, estimate,
add or subtract numbers yourself. If a figure is not in FACTS, say it was not computed.
If the user asked you to state a figure that is not in FACTS, say that the requested figure is not supported
by the plan, without repeating it.
Write 2-5 sentences, plain text, no markdown tables."""

SYSTEM_BRIEF = """You write a one-page brief for a manufacturer's leadership team.
Use ONLY numbers that appear in the FACTS JSON, copied or rounded. Never calculate new numbers.
If the user asked you to state a figure that is not in FACTS, say that the requested figure is not supported
by the plan, without repeating it.
Format: markdown with these sections: Headline (1 sentence), Recommended plan, Cost vs carbon,
Capacity risk, Orders at risk, Decisions needed. Keep it under 300 words."""


def money(v: float) -> str:
    return f"{v:,.0f}"


def _plants_line(k: dict) -> str:
    parts = []
    for p, u in k["plant_utilisation"].items():
        parts.append(f"Plant {p} {k['orders_by_plant'].get(p, 0)} orders ({u * 100:.0f}% utilised)")
    return ", ".join(parts)


def template(answer_type: str, facts: dict[str, Any]) -> str:
    """Deterministic text from facts only."""
    if answer_type in ("clarification_needed", "refused"):
        return facts.get("message", "")
    k = facts.get("kpis") or {}
    w = facts.get("weights_used") or {}
    lines: list[str] = []
    kind = facts.get("kind")

    if kind == "explain":
        e = facts["explanation"]
        o = e["order"]
        lines.append(f"Order {o['order_id']} ({o['qty']} units of {o['product']} to {o['destination']}, due {o['due']}) "
                     f"is built in Plant {e['chosen']['plant']} in week {e['chosen']['week']}: landed cost "
                     f"{money(e['chosen']['cost'])}, carbon {e['chosen']['carbon_kg']:,.1f} kg, arrives {e['chosen']['arrival']}"
                     f"{' (on time)' if e['chosen']['on_time'] else ' (late)'}.")
        for alt in e["alternatives"]:
            lines.append(f"Plant {alt['plant']}: {alt['reason']}")
        return " ".join(lines)

    if kind == "tradeoff":
        pts = facts["tradeoff"]
        base = pts[0]
        lines.append(f"The cheapest plan costs {money(base['cost'])} with {base['carbon_t']:,.1f} t of carbon "
                     f"and {base['late_orders']} late orders.")
        req = facts.get("requested_cut_pct")
        shown = [p for p in pts[1:] if req is None or p["carbon_cut_pct"] == req] or pts[1:]
        for p in shown:
            if p["feasible"]:
                lines.append(f"A {p['carbon_cut_pct']:g}% carbon cut ({p['carbon_t']:,.1f} t) costs {money(p['cost'])}, "
                             f"{money(p['extra_cost'])} more ({p['extra_cost_pct']:.2f}%), with {p['late_orders']} late orders.")
            else:
                lines.append(f"A {p['carbon_cut_pct']:g}% carbon cut is not reachable without leaving orders unserved.")
        return " ".join(lines)

    if kind == "scenario":
        c = facts["comparison"]
        lines.append(f"{facts.get('scenario_label', 'Scenario')}: the new plan costs {money(k['total_cost'])} "
                     f"({'+' if c['cost_delta'] >= 0 else ''}{money(c['cost_delta'])}, {c['cost_delta_pct']:+.2f}%), "
                     f"carbon {k['total_carbon_t']:,.1f} t ({c['carbon_delta_t']:+,.1f} t), "
                     f"{k['late_orders']} late orders ({c['late_orders_delta']:+d}), {k['unserved_orders']} unserved.")
        lines.append(f"{c['orders_moved']} orders moved plant; {c['orders_unchanged_pct']:.1f}% of the plan is unchanged.")
        if c["moved_orders"]:
            mv = ", ".join(f"{m['order_id']} {m['from']}->{m['to']}" for m in c["moved_orders"][:5])
            lines.append(f"Moves include {mv}.")
        return " ".join(lines)

    if kind == "brief":
        tp = facts.get("tradeoff", [])
        c = facts.get("capacity_event_impact")
        md = ["### Headline",
              f"Recommended plan: landed cost {money(k['total_cost'])}, carbon {k['total_carbon_t']:,.1f} t, "
              f"{k['late_orders']} late and {k['unserved_orders']} unserved orders out of {k['orders_total']}.",
              "### Recommended plan",
              f"Weights {w.get('cost', 0) * 100:.0f}% cost / {w.get('carbon', 0) * 100:.0f}% carbon. {_plants_line(k)}."]
        if tp:
            md.append("### Cost vs carbon")
            for p in tp:
                if p["feasible"] and p["carbon_cut_pct"] > 0:
                    md.append(f"- {p['carbon_cut_pct']:g}% less carbon: {money(p['extra_cost'])} more "
                              f"({p['extra_cost_pct']:.2f}%), {p['late_orders']} late orders")
        if c:
            md += ["### Capacity risk",
                   f"The capacity event changes cost by {money(c['cost_delta'])} ({c['cost_delta_pct']:+.2f}%), carbon by "
                   f"{c['carbon_delta_t']:+,.1f} t and late orders by {c['late_orders_delta']:+d}; {c['orders_moved']} orders move."]
        fl = facts.get("flagged_orders", [])
        md += ["### Orders at risk",
               f"{facts.get('flagged_count', 0)} orders are late or unserved." + (" Examples: " + ", ".join(f["order_id"] for f in fl[:5]) + "." if fl else "")]
        md += ["### Decisions needed",
               "Approve the recommended plan; decide whether to pay for a deeper carbon cut; confirm customer "
               "communication for the orders at risk."]
        return "\n".join(md)

    # allocation / cheapest / infeasible
    title = facts.get("plan_title", "Plan")
    lines.append(f"{title}: landed cost {money(k['total_cost'])}, carbon {k['total_carbon_t']:,.1f} t, "
                 f"{k['late_orders']} late orders, {k['unserved_orders']} unserved, out of {k['orders_total']} orders.")
    lines.append(_plants_line(k) + ".")
    if facts.get("flagged_count"):
        lines.append(f"{facts['flagged_count']} orders are flagged (late or unserved); see flagged_orders for "
                     f"the closest option for each.")
    return " ".join(lines)


def write(answer_type: str, facts: dict[str, Any], request: str, feedback: list[str] | None = None) -> tuple[str, dict]:
    """Returns (text, info). info: method, llm_calls, tokens."""
    info = {"method": "template", "llm_calls": 0, "tokens": 0}
    if not llm.llm_enabled() or answer_type in ("clarification_needed", "refused"):
        return template(answer_type, facts), info
    system = SYSTEM_BRIEF if facts.get("kind") == "brief" else SYSTEM_SUMMARY
    user = f"REQUEST: {request}\n\nFACTS:\n{json.dumps(facts, default=str)[:12000]}"
    if feedback:
        user += ("\n\nYour previous answer used numbers that are not in FACTS: "
                 f"{feedback}. Rewrite it using only numbers from FACTS.")
    try:
        out, tok = llm.text(system, user)
        info.update(method="llm", llm_calls=1, tokens=tok)
        return out, info
    except Exception as exc:
        info.update(method="template (LLM error)", llm_calls=1, error=str(exc)[:200])
        return template(answer_type, facts), info
