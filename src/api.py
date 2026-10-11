"""FastAPI app: the Section 6 contract.

    POST /analyze        request text (+ optional weights, scenario)
    POST /ingest         multipart: file=<csv> and the table name, given as a form field `table`
                         (or `table_name`) or as a JSON form field `meta` = {"table": "orders"}
    GET  /health         API, data store and LLM status
    GET  /trace/{run_id} full audit record of one response
    GET  /validate       dataset validation report (Section 4.5)
    POST /admin/load     load every CSV in a server-side folder (default: DATA_DIR)
"""
from __future__ import annotations

import json
import logging
import traceback
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src import graph, llm
from src.config import settings
from src.database import DATA_TABLES, get_connection, get_data_version, load_trace
from src.loader import ingest_csv, load_dir
from src.models import AnalyzeRequest, AnalyzeResponse, IngestResponse
from src.validator import validate_dataset

logging.basicConfig(level=settings.LOG_LEVEL)
log = logging.getLogger("api")

app = FastAPI(title="Multi-Plant Order Allocation", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
_LLM_STATUS: dict[str, str] = {}


@app.get("/health")
def health() -> dict:
    try:
        conn = get_connection()
        conn.execute("SELECT 1")
        rows = {t: (conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                    if conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (t,)).fetchone() else 0)
                for t in DATA_TABLES}
        db = {"status": "ok", "rows": rows, "data_version": get_data_version(conn)}
    except Exception as exc:
        db = {"status": f"error: {exc}"}
    if "llm" not in _LLM_STATUS:
        _LLM_STATUS["llm"] = llm.ping()
    ready = db.get("status") == "ok" and not _LLM_STATUS["llm"].startswith("error")
    return {"status": "ready" if ready else "degraded", "api": "ok", "data_store": db, "llm": _LLM_STATUS["llm"],
            "config": {"llm_provider": settings.LLM_PROVIDER, "llm_model": settings.LLM_MODEL,
                       "mock_llm": settings.MOCK_LLM, "solver": settings.SOLVER, "planning_start": settings.PLANNING_START}}


@app.post("/analyze", response_model=AnalyzeResponse, response_model_exclude_none=False)
def analyze(req: AnalyzeRequest):
    try:
        weights = req.weights.model_dump(exclude_none=True) if req.weights else None
        return graph.run(req.request, weights, req.scenario)
    except Exception as exc:  # never a bare 500: say what failed and where to look
        run_id = f"err-{uuid.uuid4().hex[:6]}"
        log.error("analyze failed %s: %s\n%s", run_id, exc, traceback.format_exc())
        return JSONResponse(status_code=500, content={"run_id": run_id, "answer_type": "refused",
                                                      "summary": f"Internal error while planning: {exc}",
                                                      "error": str(exc)})


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...), table: Optional[str] = Form(None),
                 table_name: Optional[str] = Form(None), meta: Optional[str] = Form(None)):
    name = table or table_name
    if not name and meta:
        try:
            m = json.loads(meta)
            name = m.get("table") or m.get("table_name") or m.get("name")
        except json.JSONDecodeError:
            name = meta.strip()
    if not name and file.filename:
        stem = file.filename.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
        name = stem if stem in DATA_TABLES else None
    if name not in DATA_TABLES:
        raise HTTPException(400, f"name the table: one of {list(DATA_TABLES)} (form field 'table' or meta JSON)")
    data = await file.read()
    return ingest_csv(data, name, get_connection())


@app.get("/trace/{run_id}")
def trace(run_id: str) -> dict:
    t = load_trace(get_connection(), run_id)
    if t is None:
        raise HTTPException(404, f"no run with id {run_id}")
    return t


@app.get("/validate")
def validate() -> dict:
    return validate_dataset(get_connection())


@app.post("/admin/load")
def admin_load(folder: Optional[str] = None) -> dict:
    res = load_dir(folder or settings.DATA_DIR, get_connection())
    return {t: {k: v for k, v in r.items() if k != "validation"} for t, r in res.items()} | \
        {"validation": validate_dataset(get_connection())}


@app.on_event("startup")
def _seed() -> None:
    """Load the bundled data pack on first start so the demo works out of the box."""
    conn = get_connection()
    if get_data_version(conn) == 0 and settings.DATA_DIR:
        try:
            load_dir(settings.DATA_DIR, conn)
            log.info("seeded database from %s", settings.DATA_DIR)
        except Exception as exc:
            log.warning("could not seed data: %s", exc)
