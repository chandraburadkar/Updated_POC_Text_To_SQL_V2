from __future__ import annotations

import os
import shutil
import threading
from typing import Optional, List

from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import Chroma

from app.rag.embeddings_factory import get_embeddings
from app.rag.schema_docs import build_schema_documents

DEFAULT_PERSIST_DIR = os.path.join("data", "chroma_schema_index")

_VS: Optional[VectorStore] = None
_LOCK = threading.Lock()


def build_schema_index(
    persist_dir: str = DEFAULT_PERSIST_DIR,
    force_rebuild: bool = False,
    schema: str = "main",
    table_prefixes: List[str] | None = None,  # e.g. ["gold_", "silver_"]
) -> VectorStore:
    """
    Builds OR loads a persistent Chroma vector index for schema RAG.

    - If persist_dir exists and not force_rebuild -> loads it
    - Else rebuilds from DuckDB information_schema and persists
    """
    embeddings = get_embeddings()
    os.makedirs(persist_dir, exist_ok=True)

    # Load if already present
    if (not force_rebuild) and os.path.isdir(persist_dir) and os.listdir(persist_dir):
        return Chroma(
            collection_name="schema_index",
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )

    # Force rebuild: delete and recreate
    if os.path.isdir(persist_dir):
        shutil.rmtree(persist_dir, ignore_errors=True)
    os.makedirs(persist_dir, exist_ok=True)

    docs = build_schema_documents(schema=schema, table_prefixes=table_prefixes)

    vs = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="schema_index",
        persist_directory=persist_dir,
    )

    # Chroma persistence is usually automatic, but keep this for older versions.
    try:
        vs.persist()
    except Exception:
        pass

    return vs


def get_schema_index(
    force_rebuild: bool = False,
    schema: str = "main",
    table_prefixes: List[str] | None = None,
) -> VectorStore:
    """
    Cached accessor: returns a single in-memory instance of the vector store.
    """
    global _VS

    if _VS is not None and not force_rebuild:
        return _VS

    with _LOCK:
        if _VS is not None and not force_rebuild:
            return _VS

        _VS = build_schema_index(
            force_rebuild=force_rebuild,
            schema=schema,
            table_prefixes=table_prefixes,
        )
        return _VS