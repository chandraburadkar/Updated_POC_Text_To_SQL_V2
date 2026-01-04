# Data & Schema Overview

This project uses DuckDB files and a schema vector index for Text2SQL grounding. Use this guide to understand what data is available and how to inspect or rebuild it.

## Assets in `data/`
- `garv.duckdb`, `amadeus_ops.duckdb`: primary analytical DuckDB databases (airline/airport ops).
- `chroma_schema_index/`: Chroma vector store files that index schema docs for RAG (`schema_index.py` builds/reads).

## How to inspect tables (DuckDB)
Activate the API/UI venv, then:
```bash
duckdb data/garv.duckdb
-- list tables
.tables
-- table schema
.schema <table_name>
-- sample rows
SELECT * FROM <table_name> LIMIT 20;
```

Or via Python REPL:
```python
from app.db.duckdb_client import get_conn
conn = get_conn()
print(conn.execute("SHOW TABLES").fetchall())
print(conn.execute("DESCRIBE main.<table>").fetchall())
print(conn.execute("SELECT * FROM main.<table> LIMIT 5").fetchdf())
```

## How the schema index is built
Pipelines under `app/pipelines/`:
- `01_create_silver_tables.py`: cleans raw data into silver tables (run first).
- `02_build_gold_tables.py`: aggregates/features for Text2SQL tasks.
- `03_build_schema_index.py`: generates textual schema docs and builds the Chroma index in `data/chroma_schema_index/`.

Run them in order (API/UI venv):
```bash
PYTHONPATH=. python app/pipelines/01_create_silver_tables.py
PYTHONPATH=. python app/pipelines/02_build_gold_tables.py
PYTHONPATH=. python app/pipelines/03_build_schema_index.py
```

## What the model sees
- RAG: `schema_index.py` retrieves schema chunks (table/column descriptions) from `chroma_schema_index` based on the rewritten query.
- SQL execution: queries are run against DuckDB (currently `data/garv.duckdb`/`amadeus_ops.duckdb`). MCP governance enforces read-only and row caps.
- Explanation: Uses returned rows/df for summaries; no direct access to raw files.

## Recommended table profiling (quick commands)
```sql
-- Row counts per table
SELECT table_name, row_count FROM duckdb_tables();
-- Column stats for a table
SELECT * FROM duckdb_columns() WHERE table_name='<table>';
-- Quick numeric profile
SELECT
  count(*) AS n,
  min(<col>) AS min_v,
  max(<col>) AS max_v,
  avg(<col>) AS avg_v
FROM <table>;
```

## Regenerating data from scratch
- Ensure raw data sources (if any) are reachable; scripts assume local files/paths referenced inside the pipeline scripts.
- Delete or move old `data/*.duckdb` and `data/chroma_schema_index/` if you want a clean rebuild.
- Rerun the three pipeline scripts in order.

## Known governance constraints (MCP)
- Only SELECT/CTE allowed (DDL/DML blocked).
- Row cap (default 500) and default LIMIT apply; use lower limits in UI/API for faster responses.
- API key + role headers required by MCP bridge (`x-mcp-api-key`, `x-mcp-role`).
