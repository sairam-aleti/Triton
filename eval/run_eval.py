"""Evaluation runner (Section 7).

    python -m eval.run_eval                 # all cases + comparisons, writes eval/report.md and eval/results.json
    python -m eval.run_eval --quick         # cases only

Method: exact match on structured fields of the response (answer_type, plant per order, flags,
validator) and an automatic number check for briefs (every number must match a tool result).
No LLM judge is used. Expected results for the tiny datasets are worked out by hand
(eval/datasets.py); pack cases check rules and properties rather than exact plans.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("PLANNING_START", "2026-10-12")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.datasets import DATASETS, load_into  # noqa: E402
from src import graph  # noqa: E402
from src.checker import validate_plan  # noqa: E402
from src.config import settings  # noqa: E402
from src.cost_model import build_problem, load_frames  # noqa: E402
from src.database import get_connection  # noqa: E402
from src.loader import load_dir  # noqa: E402
from src.solver import greedy_cheapest, solve, weight_sweep  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="eval-"))


def db_for(dataset: str, args: dict | None) -> object:
    path = TMP / f"{dataset}-{json.dumps(args or {}, sort_keys=True).replace(' ', '')}.db"
    fresh = not path.exists()
    conn = get_connection(str(path))
    if fresh:
        if dataset == "pack":
            load_dir(ROOT / "data", conn)
        else:
            load_into(conn, DATASETS[dataset](**(args or {})))
    return conn


def check(case: dict, r: dict) -> list[str]:
    """Returns the list of failed expectations (empty = pass)."""
    e, fails = case["expect"], []
    alloc = {a["order_id"]: a for a in r.get("allocation", [])}
    if "answer_type" in e and r.get("answer_type") != e["answer_type"]:
        fails.append(f"answer_type {r.get('answer_type')} != {e['answer_type']}")
    for oid, p in e.get("plant_of", {}).items():
        if alloc.get(oid, {}).get("plant") != p:
            fails.append(f"{oid} at {alloc.get(oid, {}).get('plant')} != {p}")
    for oid, p in e.get("not_plant", {}).items():
        if alloc.get(oid, {}).get("plant") == p:
            fails.append(f"{oid} should not be at {p}")
    for oid, w in e.get("week_not", {}).items():
        if alloc.get(oid, {}).get("week") == w:
            fails.append(f"{oid} built in week {w}")
    if e.get("all_on_time") and not all(a["on_time"] for a in alloc.values()):
        fails.append("some orders late")
    if "valid" in e and bool((r.get("validator") or {}).get("valid")) != e["valid"]:
        fails.append(f"validator valid={((r.get('validator') or {}).get('valid'))}")
    if "min_moved" in e and len(r.get("moved_orders", [])) < e["min_moved"]:
        fails.append(f"moved {len(r.get('moved_orders', []))} < {e['min_moved']}")
    if "no_plant" in e and any(a["plant"] == e["no_plant"] for a in alloc.values()):
        fails.append(f"orders still at {e['no_plant']}")
    unserved = [u["order_id"] for u in r.get("unserved", [])]
    for oid in e.get("unserved_includes", []):
        if oid not in unserved:
            fails.append(f"{oid} not flagged unserved")
    if "min_unserved" in e and len(unserved) < e["min_unserved"]:
        fails.append(f"unserved {len(unserved)} < {e['min_unserved']}")
    if e.get("number_check_ok") and not (r.get("number_check") or {}).get("ok"):
        fails.append(f"number check failed: {(r.get('number_check') or {}).get('unverified')}")
    for s in e.get("summary_excludes", []):
        if s.lower() in r.get("summary", "").lower():
            fails.append(f"summary repeats unsupported figure '{s}'")
    if e.get("has_explanation") and not r.get("explanation"):
        fails.append("no explanation")
    if "tradeoff_point" in e and not any(p["carbon_cut_pct"] == e["tradeoff_point"] and p["feasible"]
                                         for p in r.get("tradeoff", [])):
        fails.append(f"no feasible {e['tradeoff_point']}% point")
    return fails


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))], 1) if xs else 0.0


def run_cases() -> list[dict]:
    cases = json.loads((ROOT / "eval" / "cases.json").read_text())
    out = []
    for c in cases:
        conn = db_for(c["dataset"], c.get("args"))
        t = time.perf_counter()
        r = graph.run(c["request"], c.get("weights"), c.get("scenario"), conn=conn)
        ms = (time.perf_counter() - t) * 1000
        fails = check(c, r)
        out.append({"id": c["id"], "group": c["group"], "dataset": c["dataset"], "request": c["request"],
                    "answer_type": r.get("answer_type"), "pass": not fails, "fails": fails,
                    "latency_ms": round(ms, 1), "llm_calls": r["trace_summary"]["llm_calls"],
                    "tokens": r["trace_summary"]["tokens"], "valid": (r.get("validator") or {}).get("valid"),
                    "faithfulness": (r.get("number_check") or {}).get("faithfulness"),
                    "run_id": r["run_id"], "comparison": {k: v for k, v in (r.get("comparison") or {}).items()
                                                          if k != "moved_orders"}})
        print(f"{c['id']} {'PASS' if not fails else 'FAIL'} {c['request'][:60]:60} {ms / 1000:5.1f}s {fails}")
    return out


def comparisons() -> dict:
    conn = db_for("pack", None)
    frames = load_frames(conn)
    pdata = build_problem(frames)
    res: dict = {}
    t = time.perf_counter()
    milp = solve(pdata, label="solver default")
    t_milp = (time.perf_counter() - t) * 1000
    greedy = greedy_cheapest(pdata)
    for name, plan, ms in [("solver (CP-SAT, 0.7/0.3)", milp, t_milp), ("greedy cheapest plant", greedy, 0.0)]:
        v = validate_plan(frames, pdata, plan)
        res.setdefault("baseline", []).append({"config": name, "total_cost": plan["kpis"]["total_cost"],
                                               "total_carbon_t": plan["kpis"]["total_carbon_t"],
                                               "late_orders": plan["kpis"]["late_orders"],
                                               "unserved": plan["kpis"]["unserved_orders"],
                                               "capacity_ok": v["capacity_ok"], "valid": v["valid"],
                                               "solve_ms": round(ms, 1), "violations": len(v["violations"])})
    # Back-end comparison on the same model
    rows = []
    for backend in ("cpsat", "cbc"):
        settings.SOLVER = backend
        t = time.perf_counter()
        p = solve(pdata, label=backend)
        rows.append({"solver": backend, "objective": p["objective"], "total_cost": p["kpis"]["total_cost"],
                     "total_carbon_t": p["kpis"]["total_carbon_t"], "late_orders": p["kpis"]["late_orders"],
                     "status": p["status"], "gap": p.get("gap"), "solve_ms": round((time.perf_counter() - t) * 1000, 1)})
    settings.SOLVER = "cpsat"
    res["solver_backends"] = rows
    res["weight_sweep"] = weight_sweep(pdata, steps=6)
    return res


def write_report(cases: list[dict], comp: dict | None) -> str:
    lat = [c["latency_ms"] for c in cases]
    planned = [c for c in cases if c["valid"] is not None]
    faith = [c["faithfulness"] for c in cases if c["faithfulness"] is not None and c["answer_type"] not in ("refused", "clarification_needed")]
    lines = ["# Evaluation report", "",
             f"Config: LLM_PROVIDER={settings.LLM_PROVIDER} LLM_MODEL={settings.LLM_MODEL} MOCK_LLM={settings.MOCK_LLM} "
             f"SOLVER={settings.SOLVER} PLANNING_START={settings.PLANNING_START}", "",
             "Method: exact match on structured response fields (answer type, plant per order, flags, validator); "
             "brief faithfulness by the automatic number check (every number in the text must match a tool result). "
             "No LLM judge. Tiny datasets have hand-worked answers (eval/datasets.py).", "",
             "## Summary", "",
             "| metric | value |", "|---|---|",
             f"| cases passed | {sum(c['pass'] for c in cases)} / {len(cases)} |",
             f"| plan validity (capacity, eligibility, cost re-check) | {sum(1 for c in planned if c['valid'])} / {len(planned)} plans |",
             f"| brief/summary faithfulness (numbers matched) | {statistics.mean(faith) * 100:.1f}% over {len(faith)} answers |" if faith else "| faithfulness | n/a |",
             f"| latency p50 / p95 | {pct(lat, 0.5) / 1000:.1f}s / {pct(lat, 0.95) / 1000:.1f}s |",
             f"| LLM calls per request (mean) | {statistics.mean(c['llm_calls'] for c in cases):.2f} |",
             f"| tokens per request (mean) | {statistics.mean(c['tokens'] for c in cases):.0f} |", "",
             "## Cases", "", "| id | group | request | answer | pass | latency s | notes |", "|---|---|---|---|---|---|---|"]
    for c in cases:
        lines.append(f"| {c['id']} | {c['group']} | {c['request'][:55]} | {c['answer_type']} | "
                     f"{'yes' if c['pass'] else 'NO'} | {c['latency_ms'] / 1000:.1f} | {'; '.join(c['fails'])} |")
    rp = [c for c in cases if c["group"] == "capacity_loss" and c["comparison"]]
    if rp:
        lines += ["", "## Re-plan quality", "", "| case | cost change | carbon change (t) | late change | orders moved | unchanged |",
                  "|---|---|---|---|---|---|"]
        for c in rp:
            k = c["comparison"]
            lines.append(f"| {c['id']} | {k['cost_delta']:+,.0f} ({k['cost_delta_pct']:+.2f}%) | {k['carbon_delta_t']:+.1f} | "
                         f"{k['late_orders_delta']:+d} | {k['orders_moved']} | {k['orders_unchanged_pct']}% |")
    if comp:
        lines += ["", "## Configuration comparison 1: solver vs simple plan (data pack)", "",
                  "| config | cost | carbon t | late | unserved | capacity ok | valid | violations |", "|---|---|---|---|---|---|---|---|"]
        for b in comp["baseline"]:
            lines.append(f"| {b['config']} | {b['total_cost']:,.0f} | {b['total_carbon_t']:.1f} | {b['late_orders']} | "
                         f"{b['unserved']} | {b['capacity_ok']} | {b['valid']} | {b['violations']} |")
        lines += ["", "The greedy plan looks cheaper only because it ignores capacity: it fails validation.", "",
                  "## Configuration comparison 2: CP-SAT vs CBC on the same model", "",
                  "| solver | objective | cost | carbon t | late | status | gap | time ms |", "|---|---|---|---|---|---|---|---|"]
        for b in comp["solver_backends"]:
            lines.append(f"| {b['solver']} | {b['objective']:.4f} | {b['total_cost']:,.0f} | {b['total_carbon_t']:.1f} | "
                         f"{b['late_orders']} | {b['status']} | {b['gap']} | {b['solve_ms']:,.0f} |")
        lines += ["", "## Weight sweep (defends the 0.7/0.3 default)", "",
                  "| carbon weight | cost | carbon t | late | unserved |", "|---|---|---|---|---|"]
        for w in comp["weight_sweep"]:
            lines.append(f"| {w['w_carbon']} | {w['cost']:,.0f} | {w['carbon_t']:.1f} | {w['late_orders']} | {w['unserved_orders']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cases = run_cases()
    comp = None if args.quick else comparisons()
    report = write_report(cases, comp)
    (ROOT / "eval" / "report.md").write_text(report)
    (ROOT / "eval" / "results.json").write_text(json.dumps({"cases": cases, "comparisons": comp}, indent=2, default=str))
    print(report)


if __name__ == "__main__":
    main()
