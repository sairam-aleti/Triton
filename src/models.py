"""Pydantic v2 models: CSV row validation (Annex B) and the API contract (Section 6)."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# --------------------------------------------------------------------------------------
# API contract (Section 6)
# --------------------------------------------------------------------------------------

AnswerType = Literal["allocation", "scenario", "brief", "infeasible", "clarification_needed", "refused"]


class Weights(BaseModel):
    cost: Optional[float] = Field(default=None, ge=0)
    carbon: Optional[float] = Field(default=None, ge=0)
    lateness: Optional[float] = Field(default=None, ge=0)


class AnalyzeRequest(BaseModel):
    request: str = Field(min_length=1)
    weights: Optional[Weights] = None
    scenario: Optional[dict[str, Any]] = None


class AllocationItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    order_id: str
    plant: Optional[str]
    cost: float
    carbon_kg: float
    arrival: Optional[str]
    on_time: bool


class KPIs(BaseModel):
    model_config = ConfigDict(extra="allow")
    total_cost: float
    total_carbon_t: float
    late_orders: int
    plant_utilisation: dict[str, float]


class ValidatorResult(BaseModel):
    model_config = ConfigDict(extra="allow")
    capacity_ok: bool
    eligibility_ok: bool


class Recommendation(BaseModel):
    action: str
    rationale: str
    evidence_refs: list[str]
    confidence: Literal["high", "medium", "low"]
    estimated_impact: dict[str, float]
    requires_approval: bool


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    run_id: str
    answer_type: AnswerType
    summary: str
    allocation: list[AllocationItem] = []
    kpis: Optional[KPIs] = None
    tradeoff: list[dict[str, Any]] = []
    weights_used: dict[str, float] = {}
    validator: Optional[ValidatorResult] = None
    recommendations: list[Recommendation] = []
    assumptions: list[str] = []
    limitations: list[str] = []


class IngestResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: Literal["ok", "partial", "rejected"]
    table: str
    rows_loaded: int
    rows_rejected: int
    errors: list[dict[str, Any]] = []
    validation: Optional[dict[str, Any]] = None
