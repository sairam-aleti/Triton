"""CSV loader used by POST /ingest and by the command line (Section 4.7).

Every row is validated with the Annex B Pydantic model. Bad rows are rejected with a reason;
good rows are stored. Then the dataset-level checks (Section 4.5) run and are returned.

CLI:
    python -m src.loader data/                       # load every known CSV in a folder
    python -m src.loader --file x.csv --table orders  # load one file
"""
from __future__ import annotations

import argparse
import io
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from src.database import DATA_TABLES, get_connection, read_table, write_table
from src.models import COLUMN_ALIASES, ROW_MODELS

MAX_ERRORS_RETURNED = 50


def _normalise_headers(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Lower-case, trim, snake_case headers; map known aliases to Annex B names."""
    cols = [str(c).strip().lower().replace(" ", "_").lstrip("﻿") for c in df.columns]
    aliases = COLUMN_ALIASES.get(table, {})
    df.columns = [aliases.get(c, c) for c in cols]
    return df


def ingest_csv(file_bytes: bytes, table: str, conn: sqlite3.Connection) -> dict:
    """Validate and store one CSV. Returns rows_loaded, rows_rejected, status, errors, validation."""
    from src.validator import validate_dataset  # local import avoids a cycle at module load

    if table not in ROW_MODELS:
        return {"status": "rejected", "table": table, "rows_loaded": 0, "rows_rejected": 0,
                "errors": [{"error": f"unknown table '{table}'; use one of {list(ROW_MODELS)}"}]}
    try:
        text = file_bytes.decode("utf-8-sig")
        raw = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except Exception as exc:  # unreadable file
        return {"status": "rejected", "table": table, "rows_loaded": 0, "rows_rejected": 0,
                "errors": [{"error": f"could not read CSV: {exc}"}]}

    raw = _normalise_headers(raw, table)
    model = ROW_MODELS[table]
    required = [n for n, f in model.model_fields.items() if f.is_required()]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        return {"status": "rejected", "table": table, "rows_loaded": 0, "rows_rejected": len(raw),
                "errors": [{"error": f"missing required columns {missing}", "found": list(raw.columns)}]}

    good, errors = [], []
    for i, rec in enumerate(raw.to_dict(orient="records"), start=2):  # row 1 is the header
        rec = {k: (None if v == "" else v) for k, v in rec.items()}
        try:
            row = model.model_validate(rec)
            dumped = row.model_dump()
            good.append({k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in dumped.items()})
        except ValidationError as exc:
            first = exc.errors()[0]
            errors.append({"row": i, "field": ".".join(map(str, first["loc"])), "error": first["msg"]})

    # Duplicate keys are rejected too: they would make a plan ambiguous.
    df = pd.DataFrame(good)
    key = {"orders": ["order_id"], "plants": ["plant", "product", "week"],
           "transport_duties": ["plant", "destination"], "capacity_events": ["event_id"]}[table]
    if not df.empty:
        key = [k for k in key if k in df.columns]
        dup = df.duplicated(subset=key, keep="first")
        for idx in df.index[dup]:
            errors.append({"row": None, "field": ",".join(key), "error": f"duplicate key {df.loc[idx, key].tolist()}"})
        df = df[~dup]
    loaded = len(df)

    if not df.empty:
        if table == "capacity_events":
            # Events accumulate (the late data drop adds rows); same event_id is replaced.
            old = read_table(conn, table)
            if not old.empty:
                old = old[~old["event_id"].isin(df["event_id"])]
                df = pd.concat([old, df], ignore_index=True)
        write_table(conn, table, df, mode="replace")

    status = "ok" if not errors else ("partial" if loaded else "rejected")
    return {"status": status, "table": table, "rows_loaded": loaded, "rows_rejected": len(errors),
            "errors": errors[:MAX_ERRORS_RETURNED], "validation": validate_dataset(conn)}


def load_dir(folder: str | Path, conn: sqlite3.Connection) -> dict[str, dict]:
    """Load every Annex B file found in a folder, in dependency order."""
    out = {}
    for table in DATA_TABLES:
        path = Path(folder) / f"{table}.csv"
        if path.exists():
            out[table] = ingest_csv(path.read_bytes(), table, conn)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Load Annex B CSV files into the allocation database")
    ap.add_argument("folder", nargs="?", help="folder holding orders.csv, plants.csv, ...")
    ap.add_argument("--file", help="a single CSV file")
    ap.add_argument("--table", choices=list(ROW_MODELS), help="table name for --file")
    ap.add_argument("--db", help="database path (default: DB_PATH)")
    args = ap.parse_args(argv)
    conn = get_connection(args.db)
    if args.file:
        if not args.table:
            ap.error("--table is required with --file")
        res = {args.table: ingest_csv(Path(args.file).read_bytes(), args.table, conn)}
    elif args.folder:
        res = load_dir(args.folder, conn)
    else:
        ap.error("give a folder or --file/--table")
        return 2
    for table, r in res.items():
        print(f"[{table}] status={r['status']} loaded={r['rows_loaded']} rejected={r['rows_rejected']}")
        for e in r["errors"][:10]:
            print("   ", e)
    if res:
        from src.validator import print_report
        print_report(list(res.values())[-1]["validation"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
