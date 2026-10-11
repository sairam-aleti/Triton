"""Thin service over cost_model + solver with a cache keyed by the data version.

Identical requests on identical data return the cached plan, so a brief or a repeated what-if does
not re-solve. Any ingest bumps the data version and empties the cache.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any

from src.cost_model import ProblemData, build_problem, load_frames
from src.database import get_data_version
from src.solver import solve, tradeoff_frontier

_CACHE: dict[str, Any] = {}
_VERSION: dict[str, Any] = {"v": None}


def _key(*parts: Any) -> str:
    return json.dumps(parts, sort_keys=True, default=str)


class Planner:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        db = conn.execute("PRAGMA database_list").fetchone()[2] or f"mem:{id(conn)}"
        v = (db, get_data_version(conn))
        if v != _VERSION["v"]:
            _CACHE.clear()
            _VERSION["v"] = v
        self.version = v

    def frames(self):
        k = _key("frames")
        if k not in _CACHE:
            _CACHE[k] = load_frames(self.conn)
        return _CACHE[k]

    def problem(self, scenario: dict | None = None) -> ProblemData:
        k = _key("problem", scenario)
        if k not in _CACHE:
            _CACHE[k] = build_problem(self.frames(), scenario)
        return _CACHE[k]

    def plan(self, scenario: dict | None = None, weights: dict | None = None, *, on_time_only: bool = False,
             carbon_cap_kg: float | None = None, baseline: dict | None = None, force: dict | None = None,
             label: str = "plan") -> dict:
        k = _key("plan", scenario, weights, on_time_only, carbon_cap_kg, baseline, force)
        if k not in _CACHE:
            _CACHE[k] = solve(self.problem(scenario), weights, on_time_only=on_time_only,
                              carbon_cap_kg=carbon_cap_kg, baseline=baseline, force=force, label=label)
        plan = dict(_CACHE[k])
        plan["label"] = label
        return plan

    def frontier(self, scenario: dict | None = None, cuts: list[float] | None = None,
                 weights: dict | None = None) -> dict:
        k = _key("frontier", scenario, cuts, (weights or {}).get("lateness"))
        if k not in _CACHE:
            _CACHE[k] = tradeoff_frontier(self.problem(scenario), cuts, weights)
        return _CACHE[k]
