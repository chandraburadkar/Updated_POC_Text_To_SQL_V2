# app/agents/sql_generator.py

from __future__ import annotations


from app.audit.langsmith_tracing import traceable_fn
import logging
import re
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from app.agents.llm_factory import get_llm

logger = logging.getLogger(__name__)


# -----------------------------
# Output schema
# -----------------------------
class SQLGenOutput(BaseModel):
    sql: str = Field(..., description="DuckDB SQL query. Must be read-only SELECT.")
    used_tables: List[str] = Field(default_factory=list, description="Tables referenced in SQL.")
    used_columns: List[str] = Field(default_factory=list, description="Columns referenced in SQL (best-effort).")
    assumptions: List[str] = Field(default_factory=list, description="Assumptions made (time window, grain).")
    warnings: List[str] = Field(default_factory=list, description="Potential issues (missing filter, ambiguity).")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0..1")


def _prompt(parser: PydanticOutputParser) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a senior Data Analyst who writes SAFE DuckDB SQL for airport operations analytics.

CRITICAL RULES:
1) Output MUST be valid JSON strictly following the schema.
2) Generate READ-ONLY SQL only: SELECT queries. No INSERT/UPDATE/DELETE/CREATE/DROP/ALTER.
3) Use ONLY tables/columns that appear in the provided schema_context.
4) Always include a LIMIT for ranking / listing queries (default 50).
5) Prefer GOLD tables when available (e.g., gold_*). Use SILVER/BRONZE only if needed.
6) If time range is mentioned, apply it using the appropriate timestamp column.
   - If unsure, state assumption in 'assumptions' and use a reasonable default.
7) SQL must be executable in DuckDB.

DuckDB notes:
- Date filters: WHERE ts >= NOW() - INTERVAL '7 days'
- Use DATE_TRUNC('day', ts) or DATE_TRUNC('hour', ts) when needed
- Aggregate using AVG/SUM/COUNT; group by proper keys

Return JSON only."""
            ),
            (
                "human",
                """User question (original):
{user_question}

User question (rewritten):
{rewritten_query}

Intent:
{intent}

Entities (JSON):
{entities_json}

Schema context (authoritative; use ONLY these tables/columns):
{schema_context}

Business defaults:
- If time window missing: default to last 7 days
- If airport missing: return all airports
- For “top N”: N defaults to 5
- Always add LIMIT (top N or 50)

{format_instructions}"""
            ),
        ]
    )


# -----------------------------
# Simple guardrails / validation
# -----------------------------
_READONLY_SQL_PATTERN = re.compile(r"^\s*select\b", re.IGNORECASE)
_FORBIDDEN_PATTERN = re.compile(
    r"\b(insert|update|delete|create|drop|alter|truncate|merge|grant|revoke)\b",
    re.IGNORECASE,
)


def _basic_sql_safety_checks(sql: str) -> List[str]:
    warnings: List[str] = []
    if not _READONLY_SQL_PATTERN.search(sql or ""):
        warnings.append("SQL does not start with SELECT (must be read-only).")
    if _FORBIDDEN_PATTERN.search(sql or ""):
        warnings.append("SQL contains forbidden keywords (must be read-only).")
    if " limit " not in f" {sql.lower()} ":
        warnings.append("SQL has no LIMIT. Add LIMIT to avoid large scans.")
    return warnings


def _best_effort_extract_tables(sql: str) -> List[str]:
    # naive extraction: FROM <table> and JOIN <table>
    tables = re.findall(r"\bfrom\s+([a-zA-Z0-9_\.]+)|\bjoin\s+([a-zA-Z0-9_\.]+)", sql, flags=re.IGNORECASE)
    flat = []
    for a, b in tables:
        t = a or b
        if t:
            flat.append(t.split(".")[-1])
    # dedupe preserve order
    seen = set()
    out = []
    for t in flat:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

@traceable_fn("sql_generator")
def generate_sql(
    rewritten_query: str,
    schema_context: str,
    intent: str = "UNKNOWN",
    entities: Optional[Dict[str, Any]] = None,
    user_question: str | None = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Generate DuckDB SQL given rewritten question + schema context.
    Returns dict with sql + metadata.
    """

    entities = entities or {}
    user_question = user_question or ""

    llm = get_llm(temperature=temperature)

    parser = PydanticOutputParser(pydantic_object=SQLGenOutput)
    prompt = _prompt(parser)

    try:
        chain = prompt | llm | parser
        result: SQLGenOutput = chain.invoke(
            {
                "user_question": user_question,  # ✅ added
                "rewritten_query": rewritten_query,
                "intent": intent,
                "entities_json": json.dumps(entities, ensure_ascii=False),  # ✅ safer for LLM
                "schema_context": schema_context,
                "format_instructions": parser.get_format_instructions(),
            }
        )

        out = result.model_dump()
        out["timestamp_utc"] = datetime.utcnow().isoformat()

        # Safety checks + enrich tables
        safety_warnings = _basic_sql_safety_checks(out.get("sql", ""))
        out["warnings"] = list(dict.fromkeys(out.get("warnings", []) + safety_warnings))
        if not out.get("used_tables"):
            out["used_tables"] = _best_effort_extract_tables(out.get("sql", ""))

        return out

    except Exception as e:
        logger.error("SQL generation failed", exc_info=True)
        return {
            "sql": "",
            "used_tables": [],
            "used_columns": [],
            "assumptions": [],
            "warnings": [f"SQL generation failed: {e}"],
            "confidence": 0.0,
            "timestamp_utc": datetime.utcnow().isoformat(),
        }