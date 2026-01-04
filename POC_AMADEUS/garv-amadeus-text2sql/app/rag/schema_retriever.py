# app/rag/schema_retriever.py
from __future__ import annotations

from typing import Dict, Any, List

from app.rag.schema_index import get_schema_vectorstore


def retrieve_relevant_schema(query: str, k: int = 4) -> Dict[str, Any]:
    """
    Returns:
      - relevant_tables: list[str]
      - schema_context: concatenated text
      - rag_docs: normalized docs
    """
    vs = get_schema_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": k})

    docs = retriever.invoke(query)

    rag_docs: List[Dict[str, Any]] = []
    relevant_tables: List[str] = []

    for d in docs:
        t = d.metadata.get("table_name") or d.metadata.get("source") or "unknown"
        relevant_tables.append(t)
        rag_docs.append({"text": d.page_content, "metadata": d.metadata})

    schema_context = "\n\n".join([x["text"] for x in rag_docs])

    return {
        "relevant_tables": relevant_tables,
        "schema_context": schema_context,
        "rag_docs": rag_docs,
    }