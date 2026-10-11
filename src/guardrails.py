"""Number check (R10): every number in generated text must match a computed result.

Text numbers are compared against every numeric value in the facts JSON, allowing the usual ways
people write them: rounding to the shown decimals, thousands/millions (704.5k, 0.70M), percentages
of fractions (0.66 -> 66%), and kg <-> tonnes. Identifiers (order ids, plant letters, week numbers,
dates) are removed before checking. Anything else must match, or the text is rejected.
"""
from __future__ import annotations

import re
from typing import Any

SCALES = (1.0, 100.0, 1e-3, 1e-6, 1e3)
STRIP = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),                 # ISO dates
    re.compile(r"\b[A-Z]{1,3}-?\d+\b"),                  # ids like O-077, EV-01, S1
    re.compile(r"\bweeks?\s+\d+(\s*(-|to|and)\s*\d+)?\b", re.I),
    re.compile(r"^\s*(\d+[.)]|#+)\s", re.M),             # list markers / headings
    re.compile(r"\bR\d{1,2}\b"),                          # requirement ids
    re.compile(r"\bCO2e?\b|\bCO₂e?\b", re.I),
]
NUMBER = re.compile(r"(?<![\w.])[-+−]?\d[\d,]*(?:\.\d+)?")


def fact_numbers(facts: Any) -> list[float]:
    out: list[float] = []

    def walk(x: Any) -> None:
        if isinstance(x, bool) or x is None:
            return
        if isinstance(x, (int, float)):
            out.append(float(x))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)

    walk(facts)
    return out


def text_numbers(text: str) -> list[tuple[str, float, int]]:
    """(as written, value, decimals shown) for every number left after removing identifiers."""
    clean = text
    for rx in STRIP:
        clean = rx.sub(" ", clean)
    found = []
    for m in NUMBER.finditer(clean):
        raw = m.group(0).replace("−", "-")
        num = raw.replace(",", "")
        try:
            v = float(num)
        except ValueError:
            continue
        dec = len(num.split(".")[1]) if "." in num else 0
        found.append((raw, v, dec))
    return found


def _matches(v: float, dec: int, facts: list[float]) -> bool:
    tol_round = 0.5 * 10 ** (-dec) + 1e-9
    for f in facts:
        for s in SCALES:
            g = f * s
            if abs(abs(g) - abs(v)) <= tol_round or (v != 0 and abs(abs(g) - abs(v)) / abs(v) <= 0.005):
                return True
    return False


def check_numbers(text: str, facts: Any) -> dict[str, Any]:
    nums = fact_numbers(facts)
    found = text_numbers(text)
    bad = [raw for raw, v, dec in found if not _matches(v, dec, nums)]
    return {"ok": not bad, "numbers_checked": len(found), "unverified": bad,
            "faithfulness": round(1 - len(bad) / len(found), 4) if found else 1.0}
