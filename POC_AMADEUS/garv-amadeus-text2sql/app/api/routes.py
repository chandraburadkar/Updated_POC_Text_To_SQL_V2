# app/api/routes.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional, List
from datetime import date, datetime
from decimal import Decimal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.graph.text2sql_graph import run_text2sql

router = APIRouter()


# -----------------------------
# Helpers: JSON-safe conversion
# -----------------------------
def _json_safe(v: Any) -> Any:
    if v is None:
        return None

    if isinstance(v, (datetime, date)):
        return v.isoformat()

    if isinstance(v, Decimal):
        return float(v)

    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="ignore")

    try:
        import numpy as np  # type: ignore

        if isinstance(v, np.generic):
            return v.item()
    except Exception:
        pass

    return v


def _df_to_rows(df: Any, limit: int) -> List[Dict[str, Any]]:
    if df is None:
        return []
    try:
        rows = df.head(limit).to_dict(orient="records")
        return [{k: _json_safe(val) for k, val in r.items()} for r in rows]
    except Exception:
        return []


def _rows_json_safe(rows: Any) -> List[Dict[str, Any]]:
    """Make already-materialized rows JSON safe too."""
    if not isinstance(rows, list):
        return []
    safe: List[Dict[str, Any]] = []
    for r in rows:
        if isinstance(r, dict):
            safe.append({k: _json_safe(v) for k, v in r.items()})
    return safe


def _normalize_chat_history(chat_history: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    if not chat_history:
        return []
    out: List[Dict[str, str]] = []
    for t in chat_history[-40:]:
        role = (t.get("role") or "user").strip().lower()
        content = (t.get("content") or "").strip()
        if not content:
            continue
        out.append({"role": role, "content": content})
    return out


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


# -----------------------------
# Request/Response models
# -----------------------------
class Text2SQLRequest(BaseModel):
    question: str = Field(..., description="User natural language question")
    top_k_schema: int = Field(5, ge=1, le=20)
    return_rows: int = Field(20, ge=1, le=500)
    enable_viz: bool = Field(False, description="Chart generation hint (UI can still override)")

    chat_history: Optional[List[Dict[str, str]]] = Field(default=None)
    memory_entities: Optional[Dict[str, Any]] = Field(default=None)

    # Enterprise toggles (optional; env takes precedence if set)
    include_debug: Optional[bool] = Field(default=None, description="Return debug/trace payload")
    use_supervisor: Optional[bool] = Field(default=None, description="Use supervisor orchestration")
    role: Optional[str] = Field(default=None, description="RBAC role forwarded to MCP as x-mcp-role")


class Text2SQLResponse(BaseModel):
    ok: bool

    final_sql: Optional[str] = None
    intent: Optional[str] = None
    entities: Optional[Dict[str, Any]] = None
    retrieved_tables: Optional[List[str]] = None

    row_count: Optional[int] = None
    columns: Optional[List[str]] = None
    rows: Optional[List[Dict[str, Any]]] = None

    preview_markdown: Optional[str] = None
    explanation: Optional[Dict[str, Any]] = None

    # New: enterprise UX
    answer_card: Optional[Dict[str, Any]] = None
    chart_spec: Optional[Dict[str, Any]] = None
    suggested_questions: Optional[List[str]] = None

    chat_history: Optional[List[Dict[str, str]]] = None
    memory_entities: Optional[Dict[str, Any]] = None

    cache_hit: Optional[bool] = None
    debug: Optional[Dict[str, Any]] = None

    message: Optional[str] = None
    stage: Optional[str] = None


@router.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@router.post("/text2sql", response_model=Text2SQLResponse)
def text2sql(req: Text2SQLRequest) -> Dict[str, Any]:
    """
    Enterprise response contract:
    - Always returns answer_card + chart_spec when available (no UI hardcoding)
    - Optionally returns debug.trace if INCLUDE_DEBUG=1 or req.include_debug=true
    - Supports supervisor toggle via USE_SUPERVISOR=1 or req.use_supervisor=true
    """
    try:
        chat_history = _normalize_chat_history(req.chat_history)
        memory_entities = req.memory_entities or {}

        # toggles (env wins if present)
        include_debug = _env_bool("INCLUDE_DEBUG", False) if os.getenv("INCLUDE_DEBUG") is not None else bool(req.include_debug)
        use_supervisor = _env_bool("USE_SUPERVISOR", False) if os.getenv("USE_SUPERVISOR") is not None else bool(req.use_supervisor)

        # enable_viz: request OR env
        enable_viz = req.enable_viz or _env_bool("ENABLE_VIZ", False)

        # optional RBAC role forwarding (MCP reads x-mcp-role)
        # We set it as env for MCP client to pick up (simple, no hardcoding in code paths).
        if req.role:
            os.environ["MCP_ROLE"] = req.role

        out = run_text2sql(
            user_question=req.question,
            top_k_schema=req.top_k_schema,
            return_rows=req.return_rows,
            enable_viz=enable_viz,
            chat_history=chat_history,
            memory_entities=memory_entities,
            # NOTE: your run_text2sql will ignore these until you add supervisor mode.
            # Keeping them here makes the API stable.
            use_supervisor=use_supervisor,          # type: ignore[arg-type]
            include_debug=include_debug,            # type: ignore[arg-type]
        )

        cache_hit = bool(out.get("cache_hit", False))

        # Prepare dataframe-derived values
        df_pack = out.get("dataframe") or {}
        df = df_pack.get("df", None)

        cols: List[str] = list(df_pack.get("columns") or [])
        row_count = df_pack.get("row_count", None)

        rows: List[Dict[str, Any]] = []
        if df is not None:
            rows = _df_to_rows(df, limit=req.return_rows)
            try:
                cols = list(df.columns)
                row_count = int(df.shape[0])
            except Exception:
                pass

        # fallback if graph returns rows directly
        if not rows and isinstance(out.get("rows"), list):
            rows = _rows_json_safe(out.get("rows"))

        preview_md = out.get("preview_markdown") or df_pack.get("preview_markdown")

        # always pass through these if present
        answer_card = out.get("answer_card") or out.get("explanation", {}).get("answer_card")
        chart_spec = out.get("chart_spec") or out.get("chart") or out.get("viz")

        if out.get("ok"):
            resp: Dict[str, Any] = {
                "ok": True,
                "final_sql": out.get("final_sql"),
                "intent": out.get("intent"),
                "entities": out.get("entities"),
                "retrieved_tables": out.get("retrieved_tables"),
                "row_count": row_count,
                "columns": cols,
                "rows": rows,
                "preview_markdown": preview_md,
                "explanation": out.get("explanation"),
                "answer_card": answer_card,
                "chart_spec": chart_spec,
                "suggested_questions": out.get("suggested_questions") or [],
                "chat_history": out.get("chat_history", chat_history),
                "memory_entities": out.get("memory_entities", memory_entities),
                "cache_hit": cache_hit,
            }

            if include_debug:
                resp["debug"] = out.get("debug")
            else:
                resp["debug"] = None

            return resp

        # error / clarification
        resp_err: Dict[str, Any] = {
            "ok": False,
            "stage": out.get("stage"),
            "message": out.get("message"),
            "intent": out.get("intent"),
            "entities": out.get("entities"),
            "retrieved_tables": out.get("retrieved_tables"),
            "answer_card": answer_card,
            "chart_spec": chart_spec,
            "suggested_questions": out.get("suggested_questions") or [],
            "chat_history": out.get("chat_history", chat_history),
            "memory_entities": out.get("memory_entities", memory_entities),
            "cache_hit": cache_hit,
        }
        if include_debug:
            resp_err["debug"] = out.get("debug")
        else:
            resp_err["debug"] = None
        return resp_err

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))