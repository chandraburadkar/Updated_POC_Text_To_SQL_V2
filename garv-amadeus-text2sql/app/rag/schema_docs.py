from __future__ import annotations

from typing import List, Dict, Any
from langchain_core.documents import Document

from app.db.duckdb_client import get_conn


def _fetch_tables_and_columns(
    schema: str = "main",
    table_prefixes: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Reads DuckDB information_schema to get tables + columns.

    Optional:
      - schema: usually 'main'
      - table_prefixes: filter tables by prefixes (e.g., ["gold_", "silver_"])
    """
    conn = get_conn()

    q = """
    SELECT
        table_name,
        column_name,
        data_type,
        ordinal_position
    FROM information_schema.columns
    WHERE table_schema = ?
    ORDER BY table_name, ordinal_position
    """
    rows = conn.execute(q, [schema]).fetchall()

    tables: Dict[str, List[Dict[str, str]]] = {}
    for table_name, column_name, data_type, _ordinal in rows:
        t = str(table_name)

        # prefix filter (optional)
        if table_prefixes:
            if not any(t.startswith(p) for p in table_prefixes):
                continue

        tables.setdefault(t, []).append({"column": str(column_name), "type": str(data_type)})

    out: List[Dict[str, Any]] = []
    for t, cols in tables.items():
        out.append({"table": t, "columns": cols})

    return out


def build_schema_documents(
    schema: str = "main",
    table_prefixes: List[str] | None = None,
) -> List[Document]:
    """
    Returns schema as LangChain Documents for vector indexing.
    Each table becomes one Document.
    """
    table_specs = _fetch_tables_and_columns(schema=schema, table_prefixes=table_prefixes)

    docs: List[Document] = []
    for spec in table_specs:
        table = spec["table"]
        cols = spec["columns"]

        lines = [f"Table: {table}", "Columns:"]
        for c in cols:
            lines.append(f"- {c['column']} ({c['type']})")

        page_content = "\n".join(lines).strip()

        docs.append(
            Document(
                page_content=page_content,
                metadata={"table": table},
            )
        )

    return docs


# ✅ Backward-compat alias (if older code uses different name)
get_schema_documents = build_schema_documents