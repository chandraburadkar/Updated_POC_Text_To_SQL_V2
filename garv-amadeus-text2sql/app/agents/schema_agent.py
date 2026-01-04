from __future__ import annotations

from typing import Dict, Any, List
from app.rag.schema_index import build_schema_index


def get_schema_context(query: str, k: int = 5) -> Dict[str, Any]:
    vs = build_schema_index()
    docs = vs.similarity_search(query, k=k)

    schema_context = "\n\n".join([d.page_content for d in docs])

    return {
        "k": k,
        "schema_context": schema_context,
        "docs": [
            {"table": d.metadata.get("table"), "content": d.page_content}
            for d in docs
        ],
    }