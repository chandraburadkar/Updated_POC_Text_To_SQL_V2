# Developer Guide

This guide explains the codebase structure, how components fit together, and how to get productive quickly.

## Quickstart
1) Create venvs and install deps:
   - API/UI: `python -m venv .venv && source .venv/bin/activate && pip install -r venv_requirements.txt`
   - MCP: `python -m venv .venv_mcp && source .venv_mcp/bin/activate && pip install -r mcp_venv_requirements.txt`
2) Set `.env` (repo root): LLM_PROVIDER, GEMINI_API_KEY, GEMINI_MODEL, LANGCHAIN_API_KEY (optional), MCP_BASE_URL, MCP_API_KEY.
3) Run MCP server: `PYTHONPATH=. python -m app.mcp.server` (health: `curl http://127.0.0.1:9000/health`).
4) Run FastAPI: `PYTHONPATH=. python -m app.main --api --host 127.0.0.1 --port 8000`.
5) Run Streamlit: `PYTHONPATH=. streamlit run app/ui/streamlit_app.py --server.port 8501`.

## Architecture (current sequential pipeline)
- Entry: `app/main.py` (CLI/API); routes in `app/api/routes.py` expose `/api/text2sql`.
- UI: `app/ui/streamlit_app.py` calls API and renders rewritten query, SQL, preview, explanation.
- Pipeline orchestrator: `app/graph/text2sql_graph.py`
  1) Rewrite question → intent/entities (`app/agents/query_rewriter.py`)
  2) Schema RAG → schema_context (`app/rag/schema_index.py`, `schema_retriever.py`)
  3) SQL generation (`app/agents/sql_generator.py`)
  4) SQL validation/auto-fix (`app/agents/sql_validator.py`)
  5) Execute SQL (`app/agents/sql_executor.py`) → DuckDB/MCP bridge
  6) Explain answer (`app/agents/explainer.py`)
  7) Return stable contract: SQL, preview_markdown, explanation, debug
- State container: `app/state/agent_state.py` (permissive Pydantic model for pipeline state).
- LLM factory: `app/agents/llm_factory.py` (Gemini or Ollama based on env).
- Tracing: `app/audit/langsmith_tracing.py` (LangSmith context manager).
- Follow-up suggestions: `app/agents/followup_suggester.py`.

## MCP Layer (separate venv)
- Server: `app/mcp/server.py` (Starlette + FastMCP bridge), governance in `app/config/mcp_governance.yaml`.
- Tools/bridge endpoints: `/bridge/run_sql`, `/bridge/get_schema` (extensible to validate/profile).
- Client helpers: `app/mcp/client.py` (fastmcp) and HTTP bridge usage inside `app/agents/sql_executor.py`.
- Auth/roles: headers `x-mcp-api-key`, `x-mcp-role`; audit log JSONL.

## Data & RAG
- DuckDB client: `app/db/duckdb_client.py`.
- Pipelines: `app/pipelines/*.py` build silver/gold tables and schema index.
- Vector store: `data/chroma_schema_index/` used by `app/rag/schema_index.py`.
- See `docs/data_overview.md` for table inspection and rebuild steps.

## Common tasks
- Add a new LLM: extend `llm_factory.py` with a new provider, env flag, and necessary pip dep (API/UI venv).
- Change prompts: edit `app/config/prompts.yaml` (if present) or in-file prompts in agents.
- Adjust limits: update MCP governance (`app/config/mcp_governance.yaml`) for row caps and allowlists.
- Enable agentic supervisor (future): add `app/graph/agentic_orchestrator.py` and toggle via env (e.g., `USE_AGENTIC=1`).

## Response contract (API/UI)
`/api/text2sql` returns (success):
```
{
  "ok": true,
  "final_sql": "...",
  "preview_markdown": "...",
  "explanation": {...},
  "intent": "...",
  "entities": {...},
  "retrieved_tables": [...],
  "dataframe": {...},      # includes df/preview
  "result_df": null,       # df handle (not JSON-serializable)
  "debug": {...}
}
```
On failure: `ok=false` with `stage` and `message`, placeholders for other keys.

## Testing and verification
- Smoke: run `scripts/print_project_structure.py` and hit API/MCP health endpoints.
- Follow-up suggester: `scripts/test_text2sql_suggestions.py`.
- Manual contract check: ensure `preview_markdown` is present and `ok`/`stage` are set appropriately.

## Troubleshooting
- LLM key missing: set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) and `LLM_PROVIDER=gemini`.
- MCP auth errors: verify `MCP_API_KEY` and role header; check `mcp_governance.yaml`.
- DuckDB missing: ensure `data/*.duckdb` exists; rebuild via pipelines.
- UI shows no preview: inspect API response `stage`/`message`; verify executor returns preview_markdown.
- LangSmith errors: disable tracing envs or provide a valid key.
