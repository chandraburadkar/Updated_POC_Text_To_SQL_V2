# app/agents/sql_executor.py
from __future__ import annotations

import os
import re
import uuid
from typing import Any, Dict, Optional

import pandas as pd
import requests

from app.audit.langsmith_tracing import traceable_fn
from app.db.duckdb_client import get_conn

_LIMIT_REGEX = re.compile(r"\blimit\b", re.IGNORECASE)


# -----------------------------
# SQL helpers
# -----------------------------
def _apply_limit_if_missing(sql: str, limit: Optional[int]) -> str:
    sql_to_run = sql.strip().rstrip(";")
    if limit and limit > 0 and not _LIMIT_REGEX.search(sql_to_run):
        sql_to_run = f"{sql_to_run} LIMIT {int(limit)}"
    return sql_to_run


# -----------------------------
# MCP bridge helpers
# -----------------------------
def _mcp_base() -> str:
    return os.getenv("MCP_BASE_URL", "http://127.0.0.1:9000").rstrip("/")


def _mcp_bridge_url() -> str:
    return f"{_mcp_base()}/bridge/run_sql"


def _mcp_headers(request_id: str) -> Dict[str, str]:
    """
    Headers supported by MCP server:
      - x-mcp-api-key : auth
      - x-mcp-role    : RBAC
      - x-request-id  : audit tracing
    """
    headers: Dict[str, str] = {"x-request-id": request_id}

    api_key = os.getenv("MCP_API_KEY", "").strip()
    if api_key:
        headers["x-mcp-api-key"] = api_key

    role = os.getenv("MCP_ROLE", "analyst").strip().lower()
    if role:
        headers["x-mcp-role"] = role

    return headers


def _execute_via_mcp(sql: str, limit: int, request_id: str) -> Dict[str, Any]:
    url = _mcp_bridge_url()
    headers = _mcp_headers(request_id)

    r = requests.post(
        url,
        json={"sql": sql, "limit": int(limit)},
        headers=headers,
        timeout=60,
    )

    if not r.ok:
        try:
            err = r.json().get("error")
        except Exception:
            err = r.text
        raise RuntimeError(err)

    return r.json()


# -----------------------------
# Main executor
# -----------------------------
@traceable_fn("sql_executor")
def execute_sql(
    final_sql: str,
    limit: Optional[int] = None,
    limit_preview: int = 20,
) -> Dict[str, Any]:
    """
    MCP-first executor:
      - USE_MCP=1 → MCP HTTP bridge
      - else → local DuckDB

    Return contract (unchanged for graph):
      - df
      - preview_markdown
      - row_count
      - columns
    """
    use_mcp = os.getenv("USE_MCP", "0").lower() in ("1", "true", "yes")

    sql_to_run = _apply_limit_if_missing(final_sql, limit)
    request_id = str(uuid.uuid4())

    # -----------------------------
    # 1) MCP execution
    # -----------------------------
    mcp_err: Optional[str] = None
    if use_mcp:
        try:
            mcp_limit = int(limit) if limit else max(limit_preview, 50)

            out = _execute_via_mcp(
                sql=sql_to_run,
                limit=mcp_limit,
                request_id=request_id,
            )

            rows = out.get("rows") or []
            df = pd.DataFrame(rows)

            return {
                "row_count": int(out.get("row_count", len(df))),
                "columns": list(out.get("columns") or df.columns),
                "df": df,
                "preview_markdown": df.head(limit_preview).to_markdown(index=False) if not df.empty else "",
                "executor": "mcp",
                "request_id": request_id,
                "mcp_applied_limit": out.get("applied_limit"),
            }

        except Exception as e:
            mcp_err = str(e)

    # -----------------------------
    # 2) DuckDB fallback
    # -----------------------------
    conn = get_conn()
    df: pd.DataFrame = conn.execute(sql_to_run).df()

    return {
        "row_count": int(df.shape[0]),
        "columns": list(df.columns),
        "df": df,
        "preview_markdown": df.head(limit_preview).to_markdown(index=False) if not df.empty else "",
        "executor": "duckdb",
        "mcp_error": mcp_err,
    }