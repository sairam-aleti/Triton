"""Central configuration. Every setting comes from environment variables (.env).

Nothing here is a secret: API keys are read from the environment and never logged.
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def _bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _default_planning_start() -> str:
    """Monday on or after today. Overridden by PLANNING_START for reproducible runs."""
    today = date.today()
    return (today + timedelta(days=(7 - today.weekday()) % 7)).isoformat()


class Settings:
    # ---- LLM (switch provider with two variables, as the guide requires) ----
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").lower()  # gemini | anthropic | openai | ollama
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-3.8-flash")
    MOCK_LLM: bool = _bool("MOCK_LLM", "true")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    LLM_TIMEOUT_S: float = float(os.getenv("LLM_TIMEOUT_S", "30"))

    # ---- Storage ----
    DB_PATH: str = os.getenv("DB_PATH", str(ROOT / "allocation.db"))
    DATA_DIR: str = os.getenv("DATA_DIR", str(ROOT / "data"))

    # ---- Planning assumptions (all stated in every response's `assumptions`) ----
    PLANNING_START: str = os.getenv("PLANNING_START", "") or _default_planning_start()
    # An order produced in week w ships this many days after the week starts (Mon + 4 = Friday).
    SHIP_OFFSET_DAYS: int = int(os.getenv("SHIP_OFFSET_DAYS", "4"))
    # Extra weeks after the last due date, so a late-but-possible option always exists.
    EXTRA_WEEKS: int = int(os.getenv("EXTRA_WEEKS", "2"))

    # ---- Default weights (defended in README: knee of the frontier) ----
    DEFAULT_W_COST: float = float(os.getenv("DEFAULT_W_COST", "0.7"))
    DEFAULT_W_CARBON: float = float(os.getenv("DEFAULT_W_CARBON", "0.3"))
    DEFAULT_W_LATENESS: float = float(os.getenv("DEFAULT_W_LATENESS", "1.0"))

    # ---- Solver ----
    SOLVER: str = os.getenv("SOLVER", "cpsat")  # cpsat (OR-Tools) | cbc (PuLP)
    SOLVER_TIME_LIMIT_S: int = int(os.getenv("SOLVER_TIME_LIMIT_S", "10"))
    SOLVER_DET_TIME: float = float(os.getenv("SOLVER_DET_TIME", "0.3"))  # deterministic work units
    SOLVER_WORKERS: int = int(os.getenv("SOLVER_WORKERS", "4"))  # fixed so results do not depend on the machine
    SOLVER_GAP: float = float(os.getenv("SOLVER_GAP", "0.005"))

    # ---- Agent limits ----
    SCENARIO_MAX_TOOL_CALLS: int = int(os.getenv("SCENARIO_MAX_TOOL_CALLS", "4"))

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
