# app/db/duckdb_client.py
from __future__ import annotations

import os
import duckdb
from dotenv import load_dotenv

load_dotenv()

DEFAULT_DB_PATH = os.path.join("data", "amadeus_ops.duckdb")

def get_conn(db_path: str | None = None) -> duckdb.DuckDBPyConnection:
    path = db_path or os.getenv("DUCKDB_PATH", DEFAULT_DB_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return duckdb.connect(path)