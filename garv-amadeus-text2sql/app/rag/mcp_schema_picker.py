from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

_WORDS = re.compile(r"[a-zA-Z0-9_]+")


def _tokens(text: str) -> set[str]:
    return set(t.lower() for t in _WORDS.findall(text or ""))


def pick_top_k_tables(schema_json: Dict[str, Any], query: str, k: int = 5) -> Tuple[str, List[str]]:
    """
    Build a schema_context string + list of retrieved table names.
    Ranking is simple token overlap between query and (table + column names).
    """
    q_tokens = _tokens(query)
    tables = schema_json.get("tables") or []

    scored: List[Tuple[int, str, List[Dict[str, str]]]] = []
    for t in tables:
        tname = str(t.get("table", ""))
        cols = t.get("columns") or []
        col_names = " ".join(str(c.get("column", "")) for c in cols)
        blob = f"{tname} {col_names}"
        score = len(q_tokens.intersection(_tokens(blob)))
        scored.append((score, tname, cols))

    scored.sort(key=lambda x: x[0], reverse=True)

    # If everything scores 0, still take first k to avoid empty schema context
    top = scored[: max(1, k)]

    retrieved_tables: List[str] = []
    chunks: List[str] = []
    for _, tname, cols in top:
        retrieved_tables.append(tname)
        col_lines = ", ".join([f"{c.get('column')} ({c.get('type')})" for c in cols])
        chunks.append(f"TABLE: {tname}\nCOLUMNS: {col_lines}")

    schema_context = "\n\n".join(chunks).strip()
    return schema_context, retrieved_tables