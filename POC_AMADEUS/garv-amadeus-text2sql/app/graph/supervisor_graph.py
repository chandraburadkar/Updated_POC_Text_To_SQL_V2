# app/graph/supervisor_graph.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from app.audit.langsmith_tracing import traceable_fn, tracing_session
from app.agents.supervisor_agent import decide_next_action
from app.agents.query_rewriter import rewrite_query
from app.agents.sql_generator import generate_sql
from app.agents.sql_validator import validate_and_autofix_sql
from app.agents.sql_executor import execute_sql
from app.agents.explainer import explain_answer
from app.agents.visualizer import choose_chart_spec, render_chart
from app.mcp.bridge_client import mcp_get_schema
from app.rag.mcp_schema_picker import pick_top_k_tables


def _trace_add(trace: List[Dict[str, Any]], step: int, action: str, ok: bool, info: Dict[str, Any] | None = None):
    trace.append({"step": step, "action": action, "ok": ok, "info": info or {}})


@traceable_fn("run_text2sql_supervisor")
def run_text2sql_supervisor(
    user_question: str,
    top_k_schema: int = 5,
    return_rows: int = 20,
    chat_history: Optional[List[Dict[str, str]]] = None,
    memory_entities: Optional[Dict[str, Any]] = None,
    max_steps: int = 8,
) -> Dict[str, Any]:
    """
    Agentic supervisor loop (non-sequential):
    - supervisor decides next action
    - orchestrator enforces guardrails
    """
    with tracing_session():
        trace: List[Dict[str, Any]] = []

        state: Dict[str, Any] = {
            "user_question": user_question,
            "top_k_schema": top_k_schema,
            "return_rows": return_rows,
            "chat_history": chat_history or [],
            "memory_entities": memory_entities or {},
            # pipeline fields
            "rewritten_query": "",
            "intent": "UNKNOWN",
            "entities": {},
            "clarification_needed": False,
            "clarification_question": "",
            "schema_context": "",
            "retrieved_tables": [],
            "candidate_sql": "",
            "final_sql": "",
            "validation_ok": False,
            "fixed_by_llm": False,
            "result_df": None,
            "rows": None,
            "preview_markdown": "",
            "explanation": "",
            "answer_card": None,
            "chart_spec": None,
            "chart_path": None,
        }

        for step in range(1, max_steps + 1):
            decision = decide_next_action(state)
            action = decision.get("action")
            payload = decision.get("payload", {})

            try:
                if action == "rewrite_query":
                    out = rewrite_query(
                        user_query=state["user_question"],
                        chat_history=state["chat_history"],
                        memory_entities=state["memory_entities"],
                        temperature=0.0,
                    )
                    state["rewritten_query"] = out.get("rewritten_query") or state["user_question"]
                    state["intent"] = out.get("intent", "UNKNOWN") or "UNKNOWN"
                    state["entities"] = out.get("entities", {}) or {}
                    state["clarification_needed"] = bool(out.get("clarification_needed"))
                    state["clarification_question"] = out.get("clarification_question") or ""

                    _trace_add(trace, step, action, True, {"rewritten_query": state["rewritten_query"]})

                    if state["clarification_needed"]:
                        return {
                            "ok": False,
                            "stage": "clarification",
                            "message": state["clarification_question"] or "Need clarification.",
                            "debug": {"trace": trace},
                        }

                elif action == "fetch_schema_context":
                    # MCP schema -> pick_top_k_tables -> schema_context
                    schema_json = mcp_get_schema(table_schema="main")
                    schema_context, tables = pick_top_k_tables(
                        schema_json=schema_json,
                        query=state["rewritten_query"] or state["user_question"],
                        k=int(top_k_schema),
                    )
                    state["schema_context"] = schema_context
                    state["retrieved_tables"] = tables
                    _trace_add(trace, step, action, True, {"tables": tables})

                elif action == "generate_sql":
                    out = generate_sql(
                        rewritten_query=state["rewritten_query"] or state["user_question"],
                        schema_context=state["schema_context"],
                        intent=state["intent"],
                        entities=state["entities"],
                        user_question=state["user_question"],
                    )
                    state["candidate_sql"] = (out.get("sql") or "").strip() if isinstance(out, dict) else str(out).strip()
                    _trace_add(trace, step, action, True, {"candidate_sql_len": len(state["candidate_sql"])})

                elif action == "validate_sql":
                    if not state["candidate_sql"].strip():
                        _trace_add(trace, step, action, False, {"error": "candidate_sql missing"})
                        return {"ok": False, "message": "No SQL to validate", "debug": {"trace": trace}}

                    val = validate_and_autofix_sql(
                        rewritten_query=state["rewritten_query"] or state["user_question"],
                        schema_context=state["schema_context"],
                        candidate_sql=state["candidate_sql"],
                        max_retries=1,
                    )
                    state["validation_ok"] = bool(val.get("ok"))
                    state["final_sql"] = val.get("final_sql", state["candidate_sql"]) or state["candidate_sql"]
                    state["fixed_by_llm"] = bool(val.get("fixed_by_llm", False))

                    _trace_add(trace, step, action, True, {"validation_ok": state["validation_ok"]})

                    if not state["validation_ok"]:
                        return {
                            "ok": False,
                            "stage": "sql_validation",
                            "message": "SQL validation failed.",
                            "final_sql": state["final_sql"],
                            "debug": {"trace": trace, "validator": val},
                        }

                elif action == "execute_sql":
                    # hard guardrail
                    if not state["validation_ok"]:
                        _trace_add(trace, step, action, False, {"blocked": "execute before validation_ok"})
                        return {"ok": False, "message": "Blocked: execute before validation", "debug": {"trace": trace}}

                    exec_out = execute_sql(state["final_sql"], limit_preview=return_rows)
                    df = exec_out.get("df") if isinstance(exec_out, dict) else None

                    state["result_df"] = df
                    state["rows"] = df.to_dict("records") if isinstance(df, pd.DataFrame) else []
                    state["preview_markdown"] = exec_out.get("preview_markdown", "")

                    _trace_add(trace, step, action, True, {"rows": len(state["rows"])})

                elif action == "explain_answer":
                    state["explanation"] = explain_answer(
                        user_question=state["user_question"],
                        sql=state["final_sql"],
                        df=state["result_df"],
                    )
                    _trace_add(trace, step, action, True)

                elif action == "choose_visualization":
                    # chart_spec + chart file
                    state["chart_spec"] = choose_chart_spec(
                        user_question=state["user_question"],
                        df=state["result_df"],
                    )
                    state["chart_path"] = render_chart(
                        df=state["result_df"],
                        chart_spec=state["chart_spec"],
                    )
                    _trace_add(trace, step, action, True, {"chart_spec": state["chart_spec"]})

                elif action == "final_response":
                    # executive answer card derived from explanation + scope
                    state["answer_card"] = {
                        "answer": (state["explanation"] or "").strip()[:240],
                        "scope": {
                            "top_k_schema": top_k_schema,
                            "return_rows": return_rows,
                            "tables": state["retrieved_tables"],
                        },
                        "confidence": "High" if state["validation_ok"] else "Low",
                        "assumptions": [],
                    }

                    return {
                        "ok": True,
                        "intent": state["intent"],
                        "entities": state["entities"],
                        "rewritten_query": state["rewritten_query"],
                        "retrieved_tables": state["retrieved_tables"],
                        "candidate_sql": state["candidate_sql"],
                        "final_sql": state["final_sql"],
                        "fixed_by_llm": state["fixed_by_llm"],
                        "rows": state["rows"],
                        "preview_markdown": state["preview_markdown"],
                        "explanation": state["explanation"],
                        "answer_card": state["answer_card"],
                        "chart_spec": state["chart_spec"],
                        "chart_path": state["chart_path"],
                        "debug": {"trace": trace},
                    }

                elif action == "ask_clarification":
                    return {
                        "ok": False,
                        "stage": "clarification",
                        "message": decision.get("message") or "Need clarification.",
                        "debug": {"trace": trace},
                    }

                else:
                    _trace_add(trace, step, action or "unknown", False, {"error": "unknown action"})
                    return {"ok": False, "message": "Supervisor produced unknown action", "debug": {"trace": trace}}

            except Exception as e:
                _trace_add(trace, step, action or "unknown", False, {"exception": str(e)})
                return {"ok": False, "message": f"Error in step {action}: {e}", "debug": {"trace": trace}}

        return {"ok": False, "message": "Max steps reached without final response", "debug": {"trace": trace}}