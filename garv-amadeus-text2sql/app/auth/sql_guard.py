# app/auth/sql_guard.py
from __future__ import annotations

import re
from typing import Dict, Any, List, Set

from app.auth.policy import AccessPolicy

FORBIDDEN = re.compile(r"\b(insert|update|delete|create|drop|alter|truncate|merge|grant|revoke)\b", re.I)

# naive table extraction: FROM/JOIN tokens
TABLE_PAT = re.compile(r"\b(from|join)\s+([a-zA-Z0-9_\.]+)", re.I)


def extract_tables(sql: str) -> List[str]:
    tables = []
    for _, t in TABLE_PAT.findall(sql or ""):
        tables.append(t.split(".")[-1])
    # de-dupe preserve order
    out = []
    seen = set()
    for t in tables:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def ensure_limit(sql: str, default_limit: int = 200) -> str:
    if " limit " in f" {sql.lower()} ":
        return sql
    return sql.rstrip().rstrip(";") + f" LIMIT {default_limit};"


def guard_sql(sql: str, policy: AccessPolicy) -> Dict[str, Any]:
    """
    Returns:
      {ok: bool, final_sql: str, tables: [...], violations: [...]}
    """
    violations: List[str] = []

    if FORBIDDEN.search(sql or ""):
        violations.append("Forbidden SQL detected (write/DDL operation).")

    tables = extract_tables(sql)
    if not tables:
        violations.append("No tables detected in SQL.")
    else:
        not_allowed = [t for t in tables if t not in policy.allowed_tables]
        if not_allowed:
            violations.append(f"Access denied for tables: {not_allowed}")

    final_sql = sql
    if policy.require_limit:
        final_sql = ensure_limit(final_sql)

    return {
        "ok": len(violations) == 0,
        "final_sql": final_sql,
        "tables": tables,
        "violations": violations,
    }