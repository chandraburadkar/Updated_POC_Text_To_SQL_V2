from __future__ import annotations

import os
from typing import Any, Dict
import requests


def _mcp_base() -> str:
    return os.getenv("MCP_BASE_URL", "http://127.0.0.1:9000").rstrip("/")


def mcp_run_sql(sql: str, limit: int = 50, timeout: int = 60) -> Dict[str, Any]:
    url = f"{_mcp_base()}/bridge/run_sql"
    r = requests.post(url, json={"sql": sql, "limit": int(limit)}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def mcp_get_schema(table_schema: str = "main", timeout: int = 60) -> Dict[str, Any]:
    url = f"{_mcp_base()}/bridge/get_schema"
    r = requests.post(url, json={"table_schema": table_schema}, timeout=timeout)
    r.raise_for_status()
    return r.json()