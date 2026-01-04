# GARV Text2SQL Runbook

This document explains how to run, operate, and understand the project. It assumes two Python virtual environments:
- `.venv` for FastAPI + Streamlit (no fastmcp dependency)
- `.venv_mcp` for MCP server + bridge

## Environments and Secrets
- Create venvs:
  - API/UI: `python -m venv .venv && source .venv/bin/activate && pip install -r venv_requirements.txt`
  - MCP: `python -m venv .venv_mcp && source .venv_mcp/bin/activate && pip install -r mcp_venv_requirements.txt`
- `.env` (repo root) typical entries:
  ```
  LLM_PROVIDER=gemini
  GEMINI_API_KEY=...
  GEMINI_MODEL=gemini-2.5-flash
  LANGCHAIN_TRACING_V2=true
  LANGCHAIN_API_KEY=...
  MCP_BASE_URL=http://127.0.0.1:9000
  MCP_API_KEY=garv-secret
  ```
- Keep secrets out of git; only `.env.example` should be shared.

## How to Run
1) Start MCP server (.venv_mcp):
   ```
   cd garv-amadeus-text2sql
   source .venv_mcp/bin/activate
   PYTHONPATH=. python -m app.mcp.server
   # health: curl http://127.0.0.1:9000/health
   ```
2) Start FastAPI (.venv):
   ```
   cd garv-amadeus-text2sql
   source .venv/bin/activate
   PYTHONPATH=. python -m app.main --api --host 127.0.0.1 --port 8000
   # health: curl http://127.0.0.1:8000/api/health
   ```
3) Start Streamlit UI (.venv):
   ```
   cd garv-amadeus-text2sql
   source .venv/bin/activate
   PYTHONPATH=. streamlit run app/ui/streamlit_app.py --server.port 8501
   ```
4) One-shot CLI (no API/UI):
   ```
   cd garv-amadeus-text2sql
   source .venv/bin/activate
   PYTHONPATH=. python -m app.main "Top 5 airports by average security wait time"
   ```

## Architecture and Flow
High-level flow: Streamlit → FastAPI `/api/text2sql` → pipeline (agents + RAG) → DuckDB via MCP bridge → response with SQL + data + explanation.

Pipeline steps (sequential graph):
1) Rewrite: `app/agents/query_rewriter.py` rewrites question, extracts intent/entities.
2) Schema RAG: `app/rag/schema_index.py` / `schema_retriever.py` fetch context.
3) SQL generate: `app/agents/sql_generator.py` builds candidate SQL.
4) Validate/fix: `app/agents/sql_validator.py` checks/fixes SQL.
5) Execute: `app/agents/sql_executor.py` runs SQL (DuckDB; can call MCP bridge).
6) Explain: `app/agents/explainer.py` summarizes result.
7) Return: `app/graph/text2sql_graph.py` assembles stable response (SQL, preview_markdown, explanation, debug).

MCP server (separate venv):
- File: `app/mcp/server.py`
- Tools via HTTP bridge: `/bridge/run_sql`, `/bridge/get_schema` (extendable to validate/profile)
- Auth: headers `x-mcp-api-key`, `x-mcp-role`; governance in `app/config/mcp_governance.yaml`
- Audit: writes JSONL to configured file

UI:
- File: `app/ui/streamlit_app.py`
- Calls FastAPI `/api/text2sql`, renders rewritten query, SQL, dataframe preview, explanation; supports chat-style sessions.

State/Tracing:
- Shared state: `app/state/agent_state.py`
- LangSmith tracing: `app/audit/langsmith_tracing.py`

Data/RAG:
- DuckDB client: `app/db/duckdb_client.py`
- Pipelines: `app/pipelines/*` to build silver/gold tables and schema index
- Vector index artifacts: `data/chroma_schema_index/*`
- DuckDB files: `data/*.duckdb`

Auth/Safety:
- Placeholders in `app/auth/policy.py`, `app/auth/sql_guard.py`
- MCP governance enforces read-only and row caps

## Key Modules by Path
- Entrypoints: `app/main.py` (CLI/API), `app/api/routes.py` (routes)
- UI: `app/ui/streamlit_app.py`
- Orchestration: `app/graph/text2sql_graph.py` (sequential), planned `app/graph/agentic_orchestrator.py` (supervisor loop, if added)
- Agents: rewrite/generate/validate/execute/explain/follow-up in `app/agents/`
- RAG: `app/rag/embeddings_factory.py`, `schema_index.py`, `schema_retriever.py`, `schema_docs.py`
- MCP client/server: `app/mcp/client.py`, `app/mcp/server.py`
- Config: `app/config/llm.py` (empty stub), `app/config/mcp_governance.yaml`, `app/config/agent_tools.yaml` (if present)
- Scripts: `scripts/print_project_structure.py`, `scripts/test_text2sql_suggestions.py`
- Notebooks: `sanity_check.ipynb` (placeholder)

## API Quick Reference
- `GET /api/health` → `{status: "ok"}`
- `POST /api/text2sql` body:
  ```
  {
    "question": "...",
    "top_k_schema": 5,
    "return_rows": 20,
    "enable_viz": false
  }
  ```
  Returns `{ok, final_sql, preview_markdown, explanation, intent, entities, retrieved_tables, debug, ...}`

## MCP Quick Reference (bridge)
- Health: `curl http://127.0.0.1:9000/health`
- Run SQL: `curl -X POST -H "x-mcp-api-key: $MCP_API_KEY" -H "x-mcp-role: analyst" -H "Content-Type: application/json" -d '{"sql":"select 1 as x","limit":5}' http://127.0.0.1:9000/bridge/run_sql`
- Get schema: `curl -X POST -H "x-mcp-api-key: $MCP_API_KEY" -H "x-mcp-role: analyst" -H "Content-Type: application/json" -d '{"table_schema":"main"}' http://127.0.0.1:9000/bridge/get_schema`

## Troubleshooting
- Missing Gemini key: set `GEMINI_API_KEY` in `.env` or export it.
- MCP 401/403: ensure `x-mcp-api-key` matches governance config and role header is set.
- DuckDB errors: confirm `data/*.duckdb` exists; rebuild via pipelines if needed.
- UI empty preview: check API logs for stage-specific error; verify `preview_markdown` in response.
- LangSmith errors: unset tracing envs or set a valid `LANGCHAIN_API_KEY`.

## Evaluation Ideas
- Smoke: hit `/api/health`, `/bridge/run_sql`, and a sample `/api/text2sql`.
- Golden set: create Q→expected SQL→expected answer pairs and assert equality/tolerance.
- Contract check: ensure all responses include expected keys (`ok`, `stage/message` on failure, `final_sql`, `preview_markdown`, etc.).
- Safety: verify DDL/DML blocked by MCP governance; limits applied.
