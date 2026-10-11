"""LLM access behind one switch (LLM_PROVIDER, LLM_MODEL). MOCK_LLM=true means no LLM is called.

Every call returns its token usage so the trace can report llm_calls and tokens (R12, Section 7).
"""
from __future__ import annotations

import logging
from typing import Any

from src.config import settings

log = logging.getLogger(__name__)
_MODEL = None


def llm_enabled() -> bool:
    return not settings.MOCK_LLM


def get_model():
    """Build the chat model once. Imports are lazy so unused providers need not be installed."""
    global _MODEL
    if _MODEL is not None or settings.MOCK_LLM:
        return _MODEL
    p, m = settings.LLM_PROVIDER, settings.LLM_MODEL
    if p == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        _MODEL = ChatGoogleGenerativeAI(model=m, temperature=0, timeout=settings.LLM_TIMEOUT_S)
    elif p == "anthropic":
        from langchain_anthropic import ChatAnthropic
        _MODEL = ChatAnthropic(model=m, temperature=0, timeout=settings.LLM_TIMEOUT_S, max_tokens=2000)
    elif p == "openai":
        from langchain_openai import ChatOpenAI
        _MODEL = ChatOpenAI(model=m, temperature=0, timeout=settings.LLM_TIMEOUT_S)
    elif p == "ollama":
        from langchain_ollama import ChatOllama
        _MODEL = ChatOllama(model=m, temperature=0, base_url=settings.OLLAMA_BASE_URL)
    else:
        raise ValueError(f"unknown LLM_PROVIDER '{p}' (gemini | anthropic | openai | ollama)")
    return _MODEL


def tokens_of(msg: Any) -> int:
    usage = getattr(msg, "usage_metadata", None) or {}
    return int(usage.get("total_tokens") or (usage.get("input_tokens", 0) + usage.get("output_tokens", 0)) or 0)


def structured(system: str, user: str, schema: type) -> tuple[Any, int]:
    """Ask for JSON matching a Pydantic schema. Returns (parsed object, tokens)."""
    model = get_model().with_structured_output(schema, include_raw=True)
    out = model.invoke([("system", system), ("human", user)])
    if out.get("parsing_error") or out.get("parsed") is None:
        raise ValueError(f"structured output failed: {out.get('parsing_error')}")
    return out["parsed"], tokens_of(out.get("raw"))


def text(system: str, user: str) -> tuple[str, int]:
    msg = get_model().invoke([("system", system), ("human", user)])
    content = msg.content
    if isinstance(content, list):  # some providers return content blocks
        content = "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content).strip(), tokens_of(msg)


def ping() -> str:
    """Cheap readiness check for /health (cached by the caller)."""
    if settings.MOCK_LLM:
        return "mock"
    try:
        get_model()
        return f"configured:{settings.LLM_PROVIDER}/{settings.LLM_MODEL}"
    except Exception as exc:  # missing key or package
        return f"error: {exc}"
