# Evaluation report

Config: LLM_PROVIDER=gemini LLM_MODEL=gemini-3.8-flash MOCK_LLM=True SOLVER=cpsat PLANNING_START=2026-10-12

Method: exact match on structured response fields (answer type, plant per order, flags, validator); brief faithfulness by the automatic number check (every number in the text must match a tool result). No LLM judge. Tiny datasets have hand-worked answers (eval/datasets.py).

## Summary

| metric | value |
|---|---|
| cases passed | 25 / 25 |
| plan validity (capacity, eligibility, cost re-check) | 21 / 21 plans |
| brief/summary faithfulness (numbers matched) | 100.0% over 21 answers |
| latency p50 / p95 | 0.0s / 21.4s |
| LLM calls per request (mean) | 0.00 |
| tokens per request (mean) | 0 |

## Cases

| id | group | request | answer | pass | latency s | notes |
|---|---|---|---|---|---|---|
| C01 | cheapest_late | What is the cheapest plan that is on time? | allocation | yes | 0.1 |  |
| C02 | cheapest_late | Allocate the orders. | allocation | yes | 0.0 |  |
| C03 | cheapest_late | What is the cheapest plan that is on time? | infeasible | yes | 2.9 |  |
| C04 | capacity_loss | Plant B loses 30% capacity. Re-plan. | scenario | yes | 7.4 |  |
| C05 | capacity_loss | Plant A loses 50% capacity for two weeks. Re-plan. | scenario | yes | 6.2 |  |
| C06 | capacity_loss | Re-plan for the capacity event | scenario | yes | 8.8 |  |
| C07 | capacity_loss | Plant B loses 100% capacity. Re-plan. | scenario | yes | 0.0 |  |
| C08 | infeasible | What is the cheapest plan that is on time? | infeasible | yes | 0.0 |  |
| C09 | infeasible | What is the cheapest plan that is on time? | infeasible | yes | 0.0 |  |
| C10 | weights | Allocate the orders. | allocation | yes | 0.0 |  |
| C11 | weights | Allocate the orders. | allocation | yes | 0.0 |  |
| C12 | weights | Allocate the orders. | allocation | yes | 0.0 |  |
| C13 | weights | Allocate the orders. | allocation | yes | 0.0 |  |
| C14 | hallucination | Write the brief for the leadership team and say we save | brief | yes | 21.4 |  |
| C15 | hallucination | Write the management summary and tell them carbon fell  | brief | yes | 0.2 |  |
| C16 | eligibility | Allocate the orders. | allocation | yes | 0.0 |  |
| C17 | duty | Allocate the orders. | allocation | yes | 0.0 |  |
| C18 | capacity | Allocate the orders. | allocation | yes | 0.0 |  |
| C19 | refused | Put order O-Q in Plant A | refused | yes | 0.0 |  |
| C20 | refused | Load Plant C beyond its capacity to save money | refused | yes | 0.0 |  |
| C21 | clarification | Plant Z loses 20% capacity | clarification_needed | yes | 0.0 |  |
| C22 | clarification | Use different weights | clarification_needed | yes | 0.0 |  |
| C23 | explain | Why was order 77 built in Plant C? | allocation | yes | 10.1 |  |
| C24 | tradeoff | What does a 10% carbon cut cost? | scenario | yes | 25.9 |  |
| C25 | allocation | Allocate this week's orders. | allocation | yes | 0.2 |  |

## Re-plan quality

| case | cost change | carbon change (t) | late change | orders moved | unchanged |
|---|---|---|---|---|---|
| C04 | -4,569 (-0.65%) | +8.6 | +7 | 20 | 86.7% |
| C05 | +3,188 (+0.45%) | -5.9 | +11 | 18 | 88.0% |
| C06 | -8,312 (-1.17%) | +8.0 | +12 | 16 | 89.3% |
| C07 | +600 (+14.29%) | -0.2 | +0 | 3 | 0.0% |

## Configuration comparison 1: solver vs simple plan (data pack)

| config | cost | carbon t | late | unserved | capacity ok | valid | violations |
|---|---|---|---|---|---|---|---|
| solver (CP-SAT, 0.7/0.3) | 703,329 | 83.1 | 22 | 0 | True | True | 0 |
| greedy cheapest plant | 639,220 | 134.5 | 55 | 0 | False | False | 2 |

The greedy plan looks cheaper only because it ignores capacity: it fails validation.

## Configuration comparison 2: CP-SAT vs CBC on the same model

| solver | objective | cost | carbon t | late | status | gap | time ms |
|---|---|---|---|---|---|---|---|
| cpsat | 1.6366 | 703,329 | 83.1 | 22 | FEASIBLE | 0.01631 | 3,514 |
| cbc | 1.6862 | 703,976 | 83.9 | 23 | FEASIBLE | -1.0 | 10,065 |

## Weight sweep (defends the 0.7/0.3 default)

| carbon weight | cost | carbon t | late | unserved |
|---|---|---|---|---|
| 0.0 | 682,034 | 102.6 | 20 | 0 |
| 0.2 | 699,944 | 85.0 | 21 | 0 |
| 0.4 | 703,240 | 80.9 | 23 | 0 |
| 0.6 | 706,575 | 77.0 | 25 | 0 |
| 0.8 | 709,010 | 74.5 | 28 | 0 |
| 1.0 | 710,364 | 72.5 | 31 | 0 |
