# app/rag/embeddings_factory.py
from __future__ import annotations

from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embeddings():
    """
    Local embeddings using sentence-transformers.
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")