# Data card: synthetic multi-plant pack

| Field | Value |
|---|---|
| Name and version | synthetic-multiplant v1.0, generated 2026-10-11 |
| How it was made | `python scripts/generate_data.py --out data --seed 42 --start 2026-10-12`. Pure Python `random.Random(42)`; no LLM was used to generate rows. Plant, route and duty parameters are hand-set constants in the script. |
| What it contains | `orders.csv` 150 orders (4 products, 12 destinations, due 2026-10-20 to 2026-11-21); `plants.csv` 15 rows (3 plants x products, plus week-specific rows for plant A); `transport_duties.csv` 36 rows (3 plants x 12 destinations); `capacity_events.csv` 1 event. Expected counts are in `data_card.json` and checked by the validator. |
| Units | capacity: units/week; unit_cost, transport: currency/unit; energy: kWh/unit; carbon intensity: kg CO2e/kWh; transport carbon: kg CO2e/unit; duty_pct: percent; capacity_loss_pct: percent. `energy_price_per_kwh` and `week` are extra columns (allowed by Annex B). |
| Traps and edge cases | (1) Cheapest plant late: A is cheapest to Europe but needs 20 days; O-001..O-004 are due within 15 days. (2) Single-plant product: P4 is made only by plant B (O-007..O-009 and random P4 orders). (3) Duty: A pays 25% duty into the USA, so B is cheaper there (O-005, O-006). (4) Low carbon, little capacity: plant C (0.08 kg/kWh, 1,500 units/week). (5) Capacity loss making orders impossible: EV-01, plant B loses 60% in the first two weeks while early P4 orders depend on B. (6) Carbon by week: plant A's grid is 1.10 kg/kWh in week 2 instead of 0.80. |
| Checks run | `python -m src.validator`: all six Section 4.5 checks pass (see README for the output). |
| Known limits | Constant costs, transit times and duties across the horizon; one transport mode per route; no setup times, inventory or order splitting; destinations are countries, not addresses. |
| Privacy | Entirely synthetic. No real company, customer or personal data. Plant names are letters. |

The tiny hand-checked datasets used by the tests and the evaluation are defined in `eval/datasets.py`.
