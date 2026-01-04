# app/graph/text2sql_graph.py
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from app.audit.langsmith_tracing import tracing_session, traceable_fn
from app.services.ttl_cache import TTLCache
from app.state.agent_state import AgentState

from app.agents.query_rewriter import rewrite_query
from app.agents.sql_generator import generate_sql
from app.agents.sql_validator import validate_and_autofix_sql
from app.agents.sql_executor import execute_sql
from app.agents.explainer import explain_answer
from app.agents.llm_factory import get_llm
from app.utils.config_loader import load_yaml_config

# ✅ NEW: follow-up suggestions (LLM-based, no hardcoding)
from app.agents.followup_suggester import suggest_followup_questions

# Prefer your existing visualizer. If you later add choose_visualization, we will use it.
try:
    from app.agents.visualizer import choose_visualization  # type: ignore
except Exception:  # pragma: no cover
    choose_visualization = None

try:
    from app.agents.visualizer import suggest_plot  # type: ignore
except Exception:  # pragma: no cover
    suggest_plot = None


# -----------------------------
# CACHES
# -----------------------------
REWRITE_CACHE = TTLCache(ttl_seconds=900, max_items=500)  # 15 min
PLAN_CACHE = TTLCache(ttl_seconds=600, max_items=500)  # 10 min


# -----------------------------
# HELPERS
# -----------------------------
def _sha(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _normalize_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _chat_context_fingerprint(
    chat_history: Optional[List[Dict[str, str]]],
    memory_entities: Optional[Dict[str, Any]],
    max_turns: int = 6,
) -> str:
    """Stable small hash so follow-ups cache separately from new chats."""
    recent = (chat_history or [])[-max_turns:]
    ctx = {
        "recent": [
            {"role": (t.get("role") or ""), "content": _normalize_text(t.get("content") or "")}
            for t in recent
        ],
        "memory_entities": memory_entities or {},
    }
    return _sha(ctx)


def _safe_get_sql(candidate_sql: Any) -> str:
    if isinstance(candidate_sql, dict):
        return (candidate_sql.get("sql") or "").strip()
    if isinstance(candidate_sql, str):
        return candidate_sql.strip()
    return ""


def _as_list_unique(seq: List[str]) -> List[str]:
    return list(dict.fromkeys([s for s in seq if s]))


def _trace_add(trace: List[Dict[str, Any]], step: str, ok: bool, detail: Any = None) -> None:
    entry: Dict[str, Any] = {"step": step, "ok": bool(ok)}
    if detail is not None:
        if isinstance(detail, dict):
            # Keep trace light (never store full df/rows)
            entry["detail"] = {k: v for k, v in detail.items() if k not in ("df", "rows", "dataframe")}
        else:
            entry["detail"] = str(detail)
    trace.append(entry)


def _schema_mcp_first(rewritten_query: str, top_k_schema: int) -> Tuple[str, List[str], Dict[str, Any]]:
    """
    Returns: (schema_context, retrieved_tables, debug)
    - USE_MCP_SCHEMA=1 -> pull schema via MCP and pick top-k tables
    - else -> fallback to local schema vector index
    """
    use_mcp_schema = os.getenv("USE_MCP_SCHEMA", "0").lower() in ("1", "true", "yes")
    debug: Dict[str, Any] = {"use_mcp_schema": use_mcp_schema}

    if use_mcp_schema:
        from app.mcp.bridge_client import mcp_get_schema
        from app.rag.mcp_schema_picker import pick_top_k_tables

        schema_json = mcp_get_schema("main")
        schema_context, tables = pick_top_k_tables(
            schema_json=schema_json,
            query=rewritten_query,
            k=top_k_schema,
        )
        debug["schema_source"] = "mcp"
        debug["tables"] = tables
        return (schema_context or "").strip(), _as_list_unique(tables or []), debug

    from app.rag.schema_index import get_schema_index

    vs = get_schema_index()
    docs = vs.similarity_search(rewritten_query, k=top_k_schema)
    schema_context = "\n\n".join([(d.page_content or "") for d in docs]).strip()

    tables: List[str] = []
    for d in docs:
        if getattr(d, "metadata", None) and isinstance(d.metadata, dict):
            t = d.metadata.get("table")
            if t:
                tables.append(str(t))

    debug["schema_source"] = "rag_index"
    debug["tables"] = _as_list_unique(tables)
    return schema_context, _as_list_unique(tables), debug


def _is_chart_worthy(df: Optional[pd.DataFrame]) -> bool:
    if df is None or df.empty:
        return False
    if df.shape[0] < 2:
        return False
    return True


def _build_answer_card(
    *,
    user_question: str,
    rewritten_query: str,
    intent: str,
    entities: Dict[str, Any],
    retrieved_tables: List[str],
    explanation: Optional[Dict[str, Any]],
    executor: Optional[str],
    row_count: Optional[int],
    columns: Optional[List[str]],
    cache_hit: bool,
    fixed_by_llm: bool,
) -> Dict[str, Any]:
    """
    Enterprise “Answer Card” object for UI. No hardcoding of metrics—uses existing outputs.
    """
    summary = ""
    if isinstance(explanation, dict):
        summary = (explanation.get("summary") or "").strip()

    # lightweight confidence heuristic (non-LLM)
    confidence = "MEDIUM"
    if row_count is not None and row_count > 0 and retrieved_tables:
        confidence = "HIGH"
    if fixed_by_llm:
        confidence = "MEDIUM"  # keep conservative if LLM fixed SQL
    if row_count in (0, None):
        confidence = "LOW"

    return {
        "answer": summary or "Query executed successfully.",
        "confidence": confidence,
        "scope": {
            "intent": intent,
            "entities": entities or {},
            "tables": retrieved_tables or [],
        },
        "execution": {
            "executor": executor,
            "cache_hit": bool(cache_hit),
            "fixed_by_llm": bool(fixed_by_llm),
            "row_count": row_count,
            "columns": columns or [],
        },
        "question": user_question,
        "rewritten_query": rewritten_query,
    }


def _auto_chart_spec(
    df: Optional[pd.DataFrame],
    *,
    user_question: str,
    rewritten_query: str,
    override: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Returns a chart_spec dict (does not render the chart).
    Supports override for UI “Edit visualization”.
    """
    if df is None or df.empty:
        return None

    # Prefer a richer chooser if you have it
    if choose_visualization is not None:
        out = choose_visualization(
            df=df,
            user_question=user_question,
            rewritten_query=rewritten_query,
            override=override,
        )
        if isinstance(out, dict):
            out.setdefault("type", out.get("plot") or out.get("chart") or out.get("type"))
        return out if isinstance(out, dict) else None

    # Fallback to suggest_plot if present
    if suggest_plot is not None:
        out = suggest_plot(df)
        if isinstance(out, dict):
            chart_type = out.get("plot")
            if chart_type:
                return {
                    "type": chart_type,
                    "x": out.get("x"),
                    "y": out.get("y"),
                    "reason": out.get("reason"),
                    "override_applied": bool(override),
                }
        return None

    return None


def _df_to_rows_safe(df: Any, return_rows: int) -> List[Dict[str, Any]]:
    """Avoid pandas truthiness checks + always return list."""
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df.head(return_rows).to_dict("records")
    return []

def _suggest_next_questions(
    *,
    user_question: str,
    rewritten_query: str,
    intent: str,
    entities: Dict[str, Any],
    retrieved_tables: List[str],
    row_count: Optional[int],
    chat_history: Optional[List[Dict[str, str]]] = None,
    max_suggestions: int = 5,
) -> List[str]:
    """
    LLM-generated follow-up questions (enterprise / domain-aware).
    No hardcoding. Uses semantic layer YAML for allowed metrics/dimensions.
    """
    try:
        semantic = load_yaml_config("config/amadeus_semantic_layer.yaml")
    except Exception:
        semantic = {}

    # keep small context
    turns = (chat_history or [])[-6:]
    history_str = "\n".join([f"{t.get('role')}: {t.get('content')}" for t in turns if t.get("role") and t.get("content")]) or "None"

    import yaml as _yaml
    semantic_yaml_str = _yaml.safe_dump(semantic, sort_keys=False, allow_unicode=True)

    prompt = f"""
You are an Airport Ops analytics copilot for Amadeus-style data.
Generate {max_suggestions} short follow-up questions the user is most likely to ask next.

Rules:
- Must be relevant to the user's last question and results.
- Must be consistent with airport operations analytics & the provided semantic layer.
- No placeholders like <airport>. Use generic phrasing if airport not specified.
- Avoid repeating the same question style.
- Keep each suggestion <= 14 words.
- Output JSON ONLY: {{ "suggested_questions": ["..."] }}

Semantic layer (YAML):
{semantic_yaml_str}

Conversation (recent):
{history_str}

Last user question:
{user_question}

Rewritten query:
{rewritten_query}

Intent:
{intent}

Entities:
{json.dumps(entities or {}, ensure_ascii=False)}

Retrieved tables:
{json.dumps(retrieved_tables or [], ensure_ascii=False)}

Row count:
{row_count}
""".strip()

    try:
        llm = get_llm(temperature=0.2)
        try:
            from langchain_core.messages import HumanMessage
            raw = llm.invoke([HumanMessage(content=prompt)]).content
        except Exception:
            raw = llm.invoke(prompt).content

        # safe JSON parse
        start = (raw or "").find("{")
        end = (raw or "").rfind("}")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]
        obj = json.loads(raw)
        sugg = obj.get("suggested_questions") or []
        if not isinstance(sugg, list):
            return []
        # normalize
        clean = []
        for s in sugg:
            s = str(s).strip()
            if s and s.lower() not in [x.lower() for x in clean]:
                clean.append(s)
        return clean[:max_suggestions]
    except Exception:
        return []
    

    
# -----------------------------
# MAIN PIPELINE
# -----------------------------
@traceable_fn("run_text2sql")
def run_text2sql(
    user_question: str,
    top_k_schema: int = 5,
    return_rows: int = 20,
    enable_viz: bool = False,
    chat_history: Optional[List[Dict[str, str]]] = None,
    memory_entities: Optional[Dict[str, Any]] = None,
    viz_override: Optional[Dict[str, Any]] = None,
    include_debug: bool = True,
    use_supervisor: bool = False,
) -> Dict[str, Any]:
    """
    End-to-end Text2SQL pipeline (sequential, compatible with your current API/UI).
    """

    trace: List[Dict[str, Any]] = []
    with tracing_session():
        state = AgentState(user_question=user_question)
        state.chat_history = chat_history or []
        state.memory_entities = memory_entities or {}

        ctx_fp = _chat_context_fingerprint(state.chat_history, state.memory_entities)

        # -----------------------------
        # STEP 1: Rewrite (cached)
        # -----------------------------
        rewrite_key = _sha({"q": _normalize_text(user_question), "ctx": ctx_fp})
        rew = REWRITE_CACHE.get(rewrite_key)
        rewrite_cache_hit = rew is not None

        if not rew:
            rew = rewrite_query(
                user_query=user_question,
                chat_history=state.chat_history,
                memory_entities=state.memory_entities,
                temperature=float(os.getenv("REWRITER_TEMPERATURE", "0.0")),
            )
            REWRITE_CACHE.set(rewrite_key, rew)

        _trace_add(
            trace,
            "rewrite_query",
            ok=True,
            detail={
                "cache_hit": rewrite_cache_hit,
                "intent": rew.get("intent"),
                "clarification_needed": rew.get("clarification_needed"),
            },
        )

        if rew.get("clarification_needed"):
            answer_card = {
                "answer": rew.get("clarification_question") or "Need clarification to proceed.",
                "confidence": "LOW",
                "scope": {
                    "intent": rew.get("intent", "UNKNOWN"),
                    "entities": rew.get("entities", {}) or {},
                    "tables": [],
                },
                "execution": {"executor": None, "cache_hit": bool(rewrite_cache_hit), "fixed_by_llm": False},
                "question": user_question,
                "rewritten_query": rew.get("rewritten_query", user_question),
            }
            return {
                "ok": False,
                "stage": "clarification",
                "message": rew.get("clarification_question") or "Need clarification to proceed.",
                "intent": rew.get("intent", "UNKNOWN"),
                "rewritten_query": rew.get("rewritten_query", user_question),
                "entities": rew.get("entities", {}) or {},
                "assumptions": rew.get("assumptions", []) or [],
                "answer_card": answer_card,
                "chart_spec": None,
                "suggested_questions": [],  # ✅ NEW
                "debug": {"trace": trace} if include_debug else None,
            }

        state.rewritten_query = (rew.get("rewritten_query") or user_question).strip()
        state.intent = rew.get("intent", "UNKNOWN") or "UNKNOWN"
        state.entities = rew.get("entities", {}) or {}
        state.memory_entities = state.entities or state.memory_entities

        # -----------------------------
        # STEP 2: PLAN CACHE (reuse plan, always re-execute SQL)
        # -----------------------------
        plan_key = _sha(
            {
                "rewritten_query": _normalize_text(state.rewritten_query),
                "ctx": ctx_fp,
                "top_k_schema": int(top_k_schema),
            }
        )
        cached_plan = PLAN_CACHE.get(plan_key)
        plan_cache_hit = cached_plan is not None

        if cached_plan:
            state.final_sql = (cached_plan.get("final_sql") or "").strip()
            state.retrieved_tables = cached_plan.get("retrieved_tables", []) or []
            state.explanation = cached_plan.get("explanation")
            state.fixed_by_llm = bool(cached_plan.get("fixed_by_llm", False))
            state.schema_context = cached_plan.get("schema_context", "") or ""  # may not exist in old cache

            _trace_add(trace, "plan_cache_hit", ok=True, detail={"cache_hit": True})

            exec_out = execute_sql(state.final_sql, limit_preview=return_rows)
            df = exec_out.get("df") if isinstance(exec_out, dict) else None
            state.result_df = df

            chart_spec = None
            if enable_viz and _is_chart_worthy(df):
                try:
                    chart_spec = _auto_chart_spec(
                        df,
                        user_question=user_question,
                        rewritten_query=state.rewritten_query,
                        override=viz_override,
                    )
                    _trace_add(trace, "chart_spec", ok=True, detail={"enabled": True, "has_spec": bool(chart_spec)})
                except Exception as e:
                    _trace_add(trace, "chart_spec", ok=False, detail=str(e))

            # ✅ NEW: Suggested next questions (LLM-based)
            suggested_questions: List[str] = []
            try:
                suggested_questions = suggest_followup_questions(
                    user_question=user_question,
                    rewritten_query=state.rewritten_query,
                    intent=state.intent,
                    entities=state.entities,
                    retrieved_tables=state.retrieved_tables,
                    final_sql=state.final_sql,
                    schema_context=state.schema_context or "",
                    df=df if isinstance(df, pd.DataFrame) else None,
                    chat_history=state.chat_history,
                    temperature=float(os.getenv("FOLLOWUP_TEMPERATURE", "0.2")),
                    max_suggestions=3,
                )
                _trace_add(trace, "suggest_followups", ok=True, detail={"count": len(suggested_questions)})
            except Exception as e:
                _trace_add(trace, "suggest_followups", ok=False, detail=str(e))

            answer_card = _build_answer_card(
                user_question=user_question,
                rewritten_query=state.rewritten_query,
                intent=state.intent,
                entities=state.entities,
                retrieved_tables=state.retrieved_tables,
                explanation=state.explanation if isinstance(state.explanation, dict) else None,
                executor=exec_out.get("executor") if isinstance(exec_out, dict) else None,
                row_count=exec_out.get("row_count") if isinstance(exec_out, dict) else None,
                columns=exec_out.get("columns") if isinstance(exec_out, dict) else None,
                cache_hit=True,
                fixed_by_llm=state.fixed_by_llm,
            )

            rows = _df_to_rows_safe(df, return_rows)

            return {
                "ok": True,
                "intent": state.intent,
                "entities": state.entities,
                "rewritten_query": state.rewritten_query,
                "retrieved_tables": state.retrieved_tables,
                "final_sql": state.final_sql,
                "fixed_by_llm": state.fixed_by_llm,
                "executor": exec_out.get("executor") if isinstance(exec_out, dict) else None,
                "preview_markdown": exec_out.get("preview_markdown") if isinstance(exec_out, dict) else None,
                "rows": rows,
                "row_count": exec_out.get("row_count") if isinstance(exec_out, dict) else None,
                "columns": exec_out.get("columns") if isinstance(exec_out, dict) else None,
                "explanation": state.explanation,
                "answer_card": answer_card,
                "chart_spec": chart_spec,
                "chart": chart_spec,
                "suggested_questions": suggested_questions,  # ✅ NEW
                "cache_hit": True,
                "chat_history": state.chat_history,
                "memory_entities": state.memory_entities,
                "debug": {
                    "trace": trace,
                    "rewrite_cache_hit": rewrite_cache_hit,
                    "plan_cache_hit": True,
                    "use_supervisor": bool(use_supervisor),
                }
                if include_debug
                else None,
            }

        _trace_add(trace, "plan_cache_hit", ok=True, detail={"cache_hit": False})

        # -----------------------------
        # STEP 3: Schema retrieval (MCP-first)
        # -----------------------------
        schema_context, retrieved_tables, schema_dbg = _schema_mcp_first(state.rewritten_query, top_k_schema)
        state.schema_context = schema_context
        state.retrieved_tables = retrieved_tables

        _trace_add(trace, "schema_retrieval", ok=bool(schema_context), detail=schema_dbg)

        if not state.schema_context:
            answer_card = _build_answer_card(
                user_question=user_question,
                rewritten_query=state.rewritten_query,
                intent=state.intent,
                entities=state.entities,
                retrieved_tables=state.retrieved_tables,
                explanation=None,
                executor=None,
                row_count=None,
                columns=None,
                cache_hit=False,
                fixed_by_llm=False,
            )
            return {
                "ok": False,
                "stage": "schema",
                "message": "Schema context is empty. Build schema index or check MCP schema bridge.",
                "intent": state.intent,
                "rewritten_query": state.rewritten_query,
                "entities": state.entities,
                "retrieved_tables": state.retrieved_tables,
                "answer_card": answer_card,
                "chart_spec": None,
                "suggested_questions": [],  # ✅ NEW
                "debug": {"trace": trace} if include_debug else None,
            }

        # -----------------------------
        # STEP 4: SQL generation
        # -----------------------------
        cand = generate_sql(
            rewritten_query=state.rewritten_query,
            schema_context=state.schema_context,
            intent=state.intent,
            entities=state.entities,
            user_question=user_question,
        )
        state.candidate_sql = cand
        candidate_sql = _safe_get_sql(cand)

        _trace_add(trace, "generate_sql", ok=bool(candidate_sql), detail={"has_sql": bool(candidate_sql)})

        if not candidate_sql:
            answer_card = _build_answer_card(
                user_question=user_question,
                rewritten_query=state.rewritten_query,
                intent=state.intent,
                entities=state.entities,
                retrieved_tables=state.retrieved_tables,
                explanation=None,
                executor=None,
                row_count=None,
                columns=None,
                cache_hit=False,
                fixed_by_llm=False,
            )
            return {
                "ok": False,
                "stage": "sql_generation",
                "message": "SQL generator returned empty SQL.",
                "intent": state.intent,
                "rewritten_query": state.rewritten_query,
                "entities": state.entities,
                "retrieved_tables": state.retrieved_tables,
                "answer_card": answer_card,
                "chart_spec": None,
                "suggested_questions": [],  # ✅ NEW
                "debug": {"trace": trace} if include_debug else None,
            }

        # -----------------------------
        # STEP 5: Validation + Auto-fix
        # -----------------------------
        val = validate_and_autofix_sql(
            rewritten_query=state.rewritten_query,
            schema_context=state.schema_context,
            candidate_sql=candidate_sql,
            max_retries=1,
        )
        state.validation_ok = bool(val.get("ok", False))
        state.final_sql = (val.get("final_sql") or candidate_sql).strip()
        state.fixed_by_llm = bool(val.get("fixed_by_llm", False))

        _trace_add(
            trace,
            "validate_sql",
            ok=state.validation_ok,
            detail={
                "ok": state.validation_ok,
                "fixed_by_llm": state.fixed_by_llm,
                "error": val.get("error") or val.get("message"),
            },
        )

        if not state.validation_ok:
            answer_card = _build_answer_card(
                user_question=user_question,
                rewritten_query=state.rewritten_query,
                intent=state.intent,
                entities=state.entities,
                retrieved_tables=state.retrieved_tables,
                explanation=None,
                executor=None,
                row_count=None,
                columns=None,
                cache_hit=False,
                fixed_by_llm=state.fixed_by_llm,
            )
            return {
                "ok": False,
                "stage": "sql_validation",
                "message": "SQL validation failed.",
                "intent": state.intent,
                "rewritten_query": state.rewritten_query,
                "entities": state.entities,
                "retrieved_tables": state.retrieved_tables,
                "candidate_sql": candidate_sql,
                "final_sql": state.final_sql,
                "fixed_by_llm": state.fixed_by_llm,
                "answer_card": answer_card,
                "chart_spec": None,
                "suggested_questions": [],  # ✅ NEW
                "debug": {"trace": trace, "validator": {k: v for k, v in val.items() if k not in ("raw_prompt",)}}
                if include_debug
                else None,
            }

        # -----------------------------
        # STEP 6: Execution (MCP-aware)
        # -----------------------------
        exec_out = execute_sql(state.final_sql, limit_preview=return_rows)
        df = exec_out.get("df") if isinstance(exec_out, dict) else None
        state.result_df = df

        _trace_add(
            trace,
            "execute_sql",
            ok=True,
            detail={
                "executor": exec_out.get("executor"),
                "row_count": exec_out.get("row_count"),
                "columns": exec_out.get("columns"),
                "mcp_error": exec_out.get("mcp_error"),
            },
        )

        # -----------------------------
        # STEP 7: Explanation
        # -----------------------------
        state.explanation = explain_answer(
            user_question=user_question,
            sql=state.final_sql,
            df=state.result_df,
        )
        _trace_add(trace, "explain_answer", ok=True)

        # -----------------------------
        # STEP 8: Visualization (optional)
        # -----------------------------
        chart_spec = None
        if enable_viz and _is_chart_worthy(df):
            try:
                chart_spec = _auto_chart_spec(
                    df,
                    user_question=user_question,
                    rewritten_query=state.rewritten_query,
                    override=viz_override,
                )
                _trace_add(trace, "chart_spec", ok=True, detail={"enabled": True, "has_spec": bool(chart_spec)})
            except Exception as e:
                _trace_add(trace, "chart_spec", ok=False, detail=str(e))

        # ✅ NEW: Suggested next questions (LLM-based)  -----------------------------
        suggested_questions: List[str] = []
        try:
            suggested_questions = suggest_followup_questions(
                user_question=user_question,
                rewritten_query=state.rewritten_query,
                intent=state.intent,
                entities=state.entities,
                retrieved_tables=state.retrieved_tables,
                final_sql=state.final_sql,
                schema_context=state.schema_context or "",
                df=df if isinstance(df, pd.DataFrame) else None,
                chat_history=state.chat_history,
                temperature=float(os.getenv("FOLLOWUP_TEMPERATURE", "0.2")),
                max_suggestions=3,
            )
            _trace_add(trace, "suggest_followups", ok=True, detail={"count": len(suggested_questions)})
        except Exception as e:
            _trace_add(trace, "suggest_followups", ok=False, detail=str(e))

        # -----------------------------
        # STEP 9: Save PLAN cache (no df)
        # -----------------------------
        PLAN_CACHE.set(
            plan_key,
            {
                "final_sql": state.final_sql,
                "fixed_by_llm": state.fixed_by_llm,
                "intent": state.intent,
                "entities": state.entities,
                "retrieved_tables": state.retrieved_tables,
                "explanation": state.explanation,
                "rewritten_query": state.rewritten_query,
                # ✅ store schema_context (helps cache-hit followups be grounded)
                "schema_context": state.schema_context,
            },
        )
        _trace_add(trace, "plan_cache_set", ok=True)

        # -----------------------------
        # STEP 10: Update chat history
        # -----------------------------
        state.chat_history = (state.chat_history or []) + [
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": state.rewritten_query},
        ]

        answer_card = _build_answer_card(
            user_question=user_question,
            rewritten_query=state.rewritten_query,
            intent=state.intent,
            entities=state.entities,
            retrieved_tables=state.retrieved_tables,
            explanation=state.explanation if isinstance(state.explanation, dict) else None,
            executor=exec_out.get("executor") if isinstance(exec_out, dict) else None,
            row_count=exec_out.get("row_count") if isinstance(exec_out, dict) else None,
            columns=exec_out.get("columns") if isinstance(exec_out, dict) else None,
            cache_hit=False,
            fixed_by_llm=state.fixed_by_llm,
        )

        rows = _df_to_rows_safe(df, return_rows)

        return {
            "ok": True,
            "intent": state.intent,
            "entities": state.entities,
            "rewritten_query": state.rewritten_query,
            "retrieved_tables": state.retrieved_tables,
            "candidate_sql": candidate_sql,
            "final_sql": state.final_sql,
            "fixed_by_llm": state.fixed_by_llm,
            "executor": exec_out.get("executor") if isinstance(exec_out, dict) else None,
            "preview_markdown": exec_out.get("preview_markdown") if isinstance(exec_out, dict) else None,
            "rows": rows,
            "row_count": exec_out.get("row_count") if isinstance(exec_out, dict) else None,
            "columns": exec_out.get("columns") if isinstance(exec_out, dict) else None,
            "explanation": state.explanation,
            "answer_card": answer_card,
            "chart_spec": chart_spec,
            "chart": chart_spec,
            "suggested_questions": suggested_questions,  # ✅ NEW
            "cache_hit": False,
            "chat_history": state.chat_history,
            "memory_entities": state.memory_entities,
            "debug": {
                "trace": trace,
                "rewrite_cache_hit": rewrite_cache_hit,
                "plan_cache_hit": plan_cache_hit,
                "use_supervisor": bool(use_supervisor),
            }
            if include_debug
            else None,
        }