# app/audit/langsmith_tracing.py
from __future__ import annotations

import os
from contextlib import contextmanager

from dotenv import load_dotenv

try:
    # works with langsmith 0.2.x
    from langsmith import traceable
except Exception:
    traceable = None


def is_tracing_enabled() -> bool:
    return os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true" and bool(
        os.getenv("LANGCHAIN_API_KEY")
    )


@contextmanager
def tracing_session():
    """
    Ensures env is loaded and tracing vars exist.
    (LangChain reads env vars automatically.)
    """
    load_dotenv(override=True)
    yield


def traceable_fn(name: str):
    """
    Decorator helper to make any function show up in LangSmith traces.
    If tracing isn't installed/available, it becomes a no-op.
    """
    def decorator(fn):
        if traceable is None:
            return fn
        return traceable(name=name)(fn)
    return decorator