# app/agents/supervisor_agent.py
from __future__ import annotations

import json
from typing import Any, Dict

from app.agents.llm_factory import get_llm
from app.utils.config_loader import load_yaml_config

PROMPTS_PATH = "app/config/prompts.yaml"

ALLOWED_ACTIONS = {
    "rewrite_query",
    "fetch_schema_context",
    "generate_sql",
    "validate_sql",
    "execute_sql",
    "explain_answer",
    "choose_visualization",
    "final_response",
    "ask_clarification",
    "stop_error",
}


def _safe_json_load(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        return {}


def decide_next_action(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enterprise supervisor:
    - Emits strict JSON to choose the next tool/action based on current state.
    - Orchestrator enforces guardrails; supervisor only decides.
    """
    prompts = load_yaml_config(PROMPTS_PATH)
    sup = prompts.get("supervisor", {})
    system_prompt = sup.get("system_prompt", "")
    user_template = sup.get("user_prompt_template", "")

    # Keep the view small to reduce drift and token usage
    view = {
        "user_question": state.get("user_question", ""),
        "rewritten_query": state.get("rewritten_query", ""),
        "clarification_needed": bool(state.get("clarification_needed", False)),
        "clarification_question": state.get("clarification_question", ""),
        "schema_context_present": bool(state.get("schema_context")),
        "candidate_sql_present": bool(state.get("candidate_sql")),
        "final_sql_present": bool(state.get("final_sql")),
        "validation_ok": bool(state.get("validation_ok", False)),
        "data_present": bool(state.get("result_df") is not None) or bool(state.get("rows")),
        "explanation_present": bool(state.get("explanation")),
        "chart_spec_present": bool(state.get("chart_spec")),
        "top_k_schema": state.get("top_k_schema", 5),
        "return_rows": state.get("return_rows", 20),
    }

    user_prompt = user_template.replace("{{STATE_JSON}}", json.dumps(view, ensure_ascii=False))

    llm = get_llm(temperature=0.0)
    out = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    ).content

    j = _safe_json_load(out)
    if not j or j.get("action") not in ALLOWED_ACTIONS:
        return {
            "action": "stop_error",
            "message": "Supervisor output invalid JSON/action",
            "stop": True,
            "tool": None,
            "payload": {},
        }

    j.setdefault("payload", {})
    j.setdefault("stop", False)
    j.setdefault("message", "")
    return j