from __future__ import annotations

import os
import re
import json
import time
from typing import Any, Dict, List, Optional
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import uvicorn
import yaml
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, Mount
from starlette.requests import Request

from mcp.server.fastmcp import FastMCP
from app.db.duckdb_client import get_conn

mcp = FastMCP("garv-airops-mcp")

# -----------------------------
# YAML config loading
# -----------------------------
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"  # app/config
CONFIG_DIR = Path(os.getenv("MCP_CONFIG_DIR", str(DEFAULT_CONFIG_DIR)))

GOV_FILE = CONFIG_DIR / "mcp_governance.yaml"
RBAC_FILE = CONFIG_DIR / "mcp_rbac.yaml"

_LIMIT_REGEX = re.compile(r"\blimit\b", re.IGNORECASE)
_TABLE_REF_REGEX = re.compile(r"\bfrom\s+([a-zA-Z0-9_.]+)|\bjoin\s+([a-zA-Z0-9_.]+)", re.IGNORECASE)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_cfg() -> Dict[str, Any]:
    gov = _load_yaml(GOV_FILE)
    rbac = _load_yaml(RBAC_FILE)
    return {"gov": gov, "rbac": rbac}


# -----------------------------
# Helpers
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


def _now_iso() -> str:
    # (keeping your existing behavior; you saw the warning but it’s ok for now)
    return datetime.utcnow().isoformat() + "Z"


def _audit_log(path: str, event: Dict[str, Any]) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _extract_tables(sql: str) -> List[str]:
    tables: List[str] = []
    for m in _TABLE_REF_REGEX.finditer(sql):
        t = m.group(1) or m.group(2)
        if t:
            tables.append(t.lower())
    return list(dict.fromkeys(tables))


def _apply_limit_if_missing(sql: str, limit: int) -> str:
    sql_clean = sql.strip().rstrip(";")
    if limit > 0 and not _LIMIT_REGEX.search(sql_clean):
        sql_clean = f"{sql_clean} LIMIT {int(limit)}"
    return sql_clean


def _get_role(request: Request, rbac_cfg: Dict[str, Any]) -> str:
    role_header = (rbac_cfg.get("rbac", {}) or {}).get("role_header", "x-mcp-role")
    default_role = (rbac_cfg.get("rbac", {}) or {}).get("default_role", "analyst")
    return (request.headers.get(role_header) or default_role).strip().lower()


def _role_allowed_tables(role: str, rbac_cfg: Dict[str, Any]) -> List[str]:
    roles = rbac_cfg.get("roles", {}) or {}
    role_obj = roles.get(role, roles.get((rbac_cfg.get("rbac", {}) or {}).get("default_role", "analyst"), {})) or {}
    return [t.strip().lower() for t in (role_obj.get("tables") or [])]


def _auth_ok(request: Request, gov_cfg: Dict[str, Any]) -> bool:
    auth = gov_cfg.get("auth", {}) or {}
    enabled = bool(auth.get("enabled", False))
    if not enabled:
        return True

    header_name = auth.get("header_name", "x-mcp-api-key")
    expected = (os.getenv("MCP_API_KEY", "") or "").strip()
    if not expected:
        return False

    return (request.headers.get(header_name, "") or "").strip() == expected


def _validate_sql(
    sql: str,
    gov_cfg: Dict[str, Any],
    role_tables: List[str],
    global_allowlist: List[str],
) -> Optional[str]:
    s = (sql or "").strip()
    if not s:
        return "SQL is empty."

    forbidden = gov_cfg.get("sql_safety", {}).get("forbidden_keywords", []) or []
    forbidden_re = re.compile(r"\b(" + "|".join(map(re.escape, forbidden)) + r")\b", re.IGNORECASE) if forbidden else None
    if forbidden_re and forbidden_re.search(s):
        return "Blocked SQL: only read-only SELECT queries are allowed."

    refs = _extract_tables(s)

    if global_allowlist:
        for t in refs:
            base = t.split(".")[-1]
            if base not in global_allowlist and t not in global_allowlist:
                return f"Blocked SQL: global allowlist blocks table: {t}"

    if role_tables and "*" not in role_tables:
        for t in refs:
            base = t.split(".")[-1]
            if base not in role_tables and t not in role_tables:
                return f"Blocked SQL: role does not allow table: {t}"

    return None


def _duckdb_run_sql(sql: str, limit: int, request: Request) -> Dict[str, Any]:
    cfg = _get_cfg()
    gov = cfg["gov"]
    rbac = cfg["rbac"]

    limits = gov.get("limits", {}) or {}
    max_rows = int(limits.get("max_rows", 500))
    default_limit = int(limits.get("default_limit", 50))

    role = _get_role(request, rbac)
    role_tables = _role_allowed_tables(role, rbac)

    table_allow = gov.get("table_allowlist", {}) or {}
    global_allowlist = []
    if bool(table_allow.get("enabled", False)):
        global_allowlist = [t.strip().lower() for t in (table_allow.get("tables") or [])]

    limit = int(limit) if limit and limit > 0 else default_limit
    limit = min(limit, max_rows)

    err = _validate_sql(sql, gov, role_tables, global_allowlist)
    if err:
        raise ValueError(err)

    sql_to_run = _apply_limit_if_missing(sql, limit)
    conn = get_conn()
    df = conn.execute(sql_to_run).df()

    rows: List[Dict[str, Any]] = []
    for r in df.to_dict(orient="records"):
        rows.append({k: _json_safe(v) for k, v in r.items()})

    return {
        "row_count": int(df.shape[0]),
        "columns": list(df.columns),
        "rows": rows,
        "applied_limit": int(limit),
        "role": role,
    }


def _duckdb_get_schema(table_schema: str, request: Request) -> Dict[str, Any]:
    cfg = _get_cfg()
    rbac = cfg["rbac"]

    role = _get_role(request, rbac)
    role_tables = _role_allowed_tables(role, rbac)

    conn = get_conn()
    q = """
    SELECT table_name, column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = ?
    ORDER BY table_name, ordinal_position
    """
    rows = conn.execute(q, [table_schema]).fetchall()

    tables: Dict[str, List[Dict[str, str]]] = {}
    for table_name, column_name, data_type in rows:
        t = str(table_name).lower()
        if role_tables and "*" not in role_tables:
            if t not in role_tables:
                continue
        tables.setdefault(table_name, []).append({"column": column_name, "type": data_type})

    return {"schema": table_schema, "role": role, "tables": [{"table": t, "columns": cols} for t, cols in tables.items()]}


def _schema_context(schema_obj: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"Schema: {schema_obj.get('schema')} (role={schema_obj.get('role')})")
    for t in schema_obj.get("tables", []):
        cols = ", ".join([f"{c['column']} ({c['type']})" for c in (t.get("columns") or [])])
        lines.append(f"- {t['table']}: {cols}")
    return "\n".join(lines)


# -----------------------------
# MCP tools (same behaviour)
# -----------------------------
@mcp.tool()
def run_sql(sql: str, limit: int = 50) -> Dict[str, Any]:
    conn = get_conn()
    df = conn.execute(sql).df()
    if limit and limit > 0:
        df = df.head(int(limit))
    rows = [{k: _json_safe(v) for k, v in r.items()} for r in df.to_dict(orient="records")]
    return {"row_count": int(df.shape[0]), "columns": list(df.columns), "rows": rows}


def build_mcp_asgi_app():
    if hasattr(mcp, "http_app"):
        return mcp.http_app()
    if hasattr(mcp, "sse_app"):
        return mcp.sse_app()
    if hasattr(mcp, "app"):
        return mcp.app
    raise RuntimeError("Could not find an ASGI app on FastMCP for this MCP version.")


# -----------------------------
# Bridge endpoints (what your Text2SQL app uses)
# -----------------------------
def health(request: Request):
    cfg = _get_cfg()
    gov = cfg["gov"]
    rbac = cfg["rbac"]
    limits = gov.get("limits", {}) or {}
    auth = gov.get("auth", {}) or {}
    audit = gov.get("audit", {}) or {}

    return JSONResponse(
        {
            "status": "ok",
            "message": "GARV MCP server is running",
            "mcp_base": "/mcp",
            "bridge": {
                "run_sql": "/bridge/run_sql",
                "get_schema": "/bridge/get_schema",
                "schema_context": "/bridge/schema_context",
            },
            "governance": {
                "auth_enabled": bool(auth.get("enabled", False)),
                "max_rows": int(limits.get("max_rows", 500)),
                "default_limit": int(limits.get("default_limit", 50)),
                "audit_enabled": bool(audit.get("enabled", False)),
                "audit_path": audit.get("path"),
            },
            "rbac": {
                "default_role": (rbac.get("rbac", {}) or {}).get("default_role", "analyst"),
                "role_header": (rbac.get("rbac", {}) or {}).get("role_header", "x-mcp-role"),
                "roles": list((rbac.get("roles", {}) or {}).keys()),
            },
        }
    )


async def bridge_run_sql(request: Request):
    """
    ✅ UPDATED:
    - Accepts sql OR final_sql OR query
    - Accepts limit OR limit_preview
    - Keeps governance/validation same
    """
    started = time.time()
    cfg = _get_cfg()
    gov = cfg["gov"]
    audit_cfg = gov.get("audit", {}) or {}
    audit_enabled = bool(audit_cfg.get("enabled", False))
    audit_path = str(audit_cfg.get("path", "./mcp_audit.log"))

    req_id = request.headers.get("x-request-id", "")
    role = _get_role(request, cfg["rbac"])

    if not _auth_ok(request, gov):
        if audit_enabled:
            _audit_log(
                audit_path,
                {
                    "ts": _now_iso(),
                    "event": "bridge_run_sql",
                    "status": "DENY",
                    "reason": "bad_api_key",
                    "request_id": req_id,
                    "role": role,
                },
            )
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        body = await request.json()

        # ✅ accept multiple payload shapes (your test proved clients send these variants)
        sql = (
            str(body.get("sql", "") or "").strip()
            or str(body.get("final_sql", "") or "").strip()
            or str(body.get("query", "") or "").strip()
        )

        # ✅ accept multiple limit keys
        raw_limit = body.get("limit", None)
        if raw_limit is None:
            raw_limit = body.get("limit_preview", None)

        if raw_limit is None:
            raw_limit = (gov.get("limits", {}) or {}).get("default_limit", 50)

        limit = int(raw_limit)

        out = _duckdb_run_sql(sql=sql, limit=limit, request=request)

        if audit_enabled:
            _audit_log(
                audit_path,
                {
                    "ts": _now_iso(),
                    "event": "bridge_run_sql",
                    "status": "OK",
                    "request_id": req_id,
                    "role": role,
                    "row_count": out.get("row_count"),
                    "applied_limit": out.get("applied_limit"),
                    "tables": _extract_tables(sql),
                    "latency_ms": int((time.time() - started) * 1000),
                },
            )
        return JSONResponse(out)

    except Exception as e:
        if audit_enabled:
            _audit_log(
                audit_path,
                {
                    "ts": _now_iso(),
                    "event": "bridge_run_sql",
                    "status": "ERROR",
                    "request_id": req_id,
                    "role": role,
                    "error": str(e),
                    "latency_ms": int((time.time() - started) * 1000),
                },
            )
        return JSONResponse({"error": str(e)}, status_code=400)


async def bridge_get_schema(request: Request):
    cfg = _get_cfg()
    gov = cfg["gov"]

    if not _auth_ok(request, gov):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        body = await request.json()
        table_schema = str(body.get("table_schema", "main"))
        out = _duckdb_get_schema(table_schema=table_schema, request=request)
        return JSONResponse(out)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


async def bridge_schema_context(request: Request):
    cfg = _get_cfg()
    gov = cfg["gov"]

    if not _auth_ok(request, gov):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        body = await request.json()
        table_schema = str(body.get("table_schema", "main"))
        schema_obj = _duckdb_get_schema(table_schema=table_schema, request=request)
        return JSONResponse({"context": _schema_context(schema_obj), "schema": schema_obj})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


def build_app() -> Starlette:
    mcp_app = build_mcp_asgi_app()
    return Starlette(
        debug=False,
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/bridge/run_sql", bridge_run_sql, methods=["POST"]),
            Route("/bridge/get_schema", bridge_get_schema, methods=["POST"]),
            Route("/bridge/schema_context", bridge_schema_context, methods=["POST"]),
            Mount("/mcp", app=mcp_app),
        ],
    )


if __name__ == "__main__":
    host = os.getenv("HOST") or os.getenv("MCP_HOST") or "127.0.0.1"
    port = int(os.getenv("PORT") or os.getenv("MCP_PORT") or "9000")

    app = build_app()
    print(f"✅ Starting MCP server on http://{host}:{port}")
    print("✅ Health:  /health")
    print("✅ Bridge:  /bridge/run_sql  /bridge/get_schema  /bridge/schema_context")
    uvicorn.run(app, host=host, port=port, log_level="info")