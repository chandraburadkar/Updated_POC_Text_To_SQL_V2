# app/agents/sql_validator.py

from __future__ import annotations

from app.audit.langsmith_tracing import traceable_fn
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from app.db.duckdb_client import get_conn
from app.agents.llm_factory import get_llm

logger = logging.getLogger(__name__)


class SQLFixOutput(BaseModel):
    fixed_sql: str = Field(..., description="Corrected SQL query for DuckDB.")
    notes: str = Field(..., description="Short explanation of what was changed.")


def validate_sql_duckdb(sql: str) -> Dict[str, Any]:
    """
    Validate SQL without executing it using DuckDB EXPLAIN.
    Returns: {ok: bool, error: str}
    """
    conn = get_conn()
    try:
        # EXPLAIN validates parsing + bindings (tables/columns), without running query
        conn.execute("EXPLAIN " + sql)
        return {"ok": True, "error": ""}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _fix_prompt(parser: PydanticOutputParser) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system",
         """You are a senior SQL engineer.
Fix the SQL for DuckDB using ONLY the available schema context.
Rules:
- Use ONLY tables/columns that exist in schema context.
- Keep the query minimal and correct.
- If date filtering is needed, use DuckDB syntax.
Return STRICT JSON matching the schema.
"""),
        ("human",
         "User request (rewritten):\n{rewritten_query}\n\n"
         "Schema context:\n{schema_context}\n\n"
         "SQL that failed:\n{bad_sql}\n\n"
         "DuckDB error:\n{duckdb_error}\n\n"
         "{format_instructions}"
         )
    ])


def fix_sql_with_llm(
    rewritten_query: str,
    schema_context: str,
    bad_sql: str,
    duckdb_error: str,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Uses the configured LLM (Ollama or Gemini) to fix SQL.
    """
    llm = get_llm(temperature=temperature)

    parser = PydanticOutputParser(pydantic_object=SQLFixOutput)
    prompt = _fix_prompt(parser)

    chain = prompt | llm | parser
    result: SQLFixOutput = chain.invoke({
        "rewritten_query": rewritten_query,
        "schema_context": schema_context,
        "bad_sql": bad_sql,
        "duckdb_error": duckdb_error,
        "format_instructions": parser.get_format_instructions(),
    })

    out = result.model_dump()
    out["timestamp_utc"] = datetime.utcnow().isoformat()
    return out

@traceable_fn("sql_validator")
def validate_and_autofix_sql(
    rewritten_query: str,
    schema_context: str,
    candidate_sql: str,
    max_retries: int = 1,
) -> Dict[str, Any]:
    """
    Validate SQL. If invalid, call LLM to fix (max_retries times).
    """
    attempt = 0
    current_sql = candidate_sql
    first_error: Optional[str] = None

    while attempt <= max_retries:
        v = validate_sql_duckdb(current_sql)
        if v["ok"]:
            return {
                "ok": True,
                "final_sql": current_sql,
                "fixed_by_llm": attempt > 0,
                "error_before_fix": first_error or "",
            }

        # failed
        if first_error is None:
            first_error = v["error"]

        if attempt == max_retries:
            return {
                "ok": False,
                "final_sql": current_sql,
                "fixed_by_llm": attempt > 0,
                "error_before_fix": first_error or "",
                "last_error": v["error"],
            }

        # ask LLM to fix
        fix = fix_sql_with_llm(
            rewritten_query=rewritten_query,
            schema_context=schema_context,
            bad_sql=current_sql,
            duckdb_error=v["error"],
            temperature=0.0,
        )
        current_sql = fix["fixed_sql"]
        attempt += 1

    # should never reach
    return {"ok": False, "final_sql": current_sql, "fixed_by_llm": False, "error_before_fix": first_error or ""}

# --- Public alias expected by the graph runner ---
def validate_sql(
    rewritten_query: str,
    schema_context: str,
    candidate_sql: str,
    max_retries: int = 1,
) -> Dict[str, Any]:
    """
    Alias for validate_and_autofix_sql so other modules can import validate_sql().
    """
    return validate_and_autofix_sql(
        rewritten_query=rewritten_query,
        schema_context=schema_context,
        candidate_sql=candidate_sql,
        max_retries=max_retries,
    )