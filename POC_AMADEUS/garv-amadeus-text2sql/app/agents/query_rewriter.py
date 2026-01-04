# app/agents/query_rewriter.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from app.agents.llm_factory import get_llm
from app.utils.config_loader import load_yaml_config


def _format_chat_history(chat_history: Optional[List[Dict[str, str]]], max_turns: int = 6) -> str:
    turns = (chat_history or [])[-max_turns:]
    lines = []
    for t in turns:
        role = (t.get("role") or "").strip()
        content = (t.get("content") or "").strip()
        if role and content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "None"


def _normalize_text(text: str, normalization_map: Dict[str, str]) -> str:
    s = (text or "").strip()
    for k, v in (normalization_map or {}).items():
        s = re.sub(rf"\b{re.escape(k)}\b", v, s, flags=re.IGNORECASE)
    s = " ".join(s.split())
    return s


def _safe_json_load(s: str) -> Dict[str, Any]:
    """Extract first JSON object from model output safely."""
    s = (s or "").strip()
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        s = s[start : end + 1]
    return json.loads(s)


def _boolish(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "yes", "y")
    return False


def rewrite_query(
    user_query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    memory_entities: Optional[Dict[str, Any]] = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Enterprise-grade Amadeus Airport Ops query rewriter.
    - Config-driven prompts + semantic layer (YAML)
    - Produces structured JSON (intent, entities, rewritten_query)
    - Asks max ONE clarification question only if truly required
    - Prefer assumptions + defaults to avoid "cheap" clarification stops
    """

    prompts = load_yaml_config("config/prompts.yaml")
    semantic = load_yaml_config("config/amadeus_semantic_layer.yaml")

    defaults = semantic.get("defaults", {}) or {}
    norm_map = semantic.get("normalization_map", {}) or {}

    # Defaults used for "smart assumptions"
    timeframe_days = int(defaults.get("timeframe_days", 30))
    top_n = int(defaults.get("top_n", 5))
    dimension = str(defaults.get("dimension", "airport"))

    # ✅ These two remove the screenshot-style clarification
    default_wait_metric = str(defaults.get("default_wait_time_metric", "security_wait_time_avg"))
    default_baseline_metric = str(defaults.get("default_comparison_baseline_metric", "overall_wait_time_avg"))

    policy = semantic.get("clarification_policy", {}) or {}
    max_q = int(policy.get("max_questions", 1))
    prefer_assumptions = bool(policy.get("prefer_assumptions", True))
    clarify_only_if_unmappable = bool(policy.get("clarify_only_if_unmappable", True))

    user_query_n = _normalize_text(user_query, norm_map)
    chat_str = _format_chat_history(chat_history)

    sys_prompt = prompts["rewriter"]["system"]
    user_template = prompts["rewriter"]["user_template"]

    # pass semantic layer raw YAML string into prompt (auditable + editable)
    import yaml as _yaml

    semantic_yaml_str = _yaml.safe_dump(semantic, sort_keys=False, allow_unicode=True)

    user_prompt = user_template.format(
        semantic_layer_yaml=semantic_yaml_str,
        timeframe_days=timeframe_days,
        top_n=top_n,
        dimension=dimension,
        chat_history=chat_str,
        user_question=user_query_n,
    )

    llm = get_llm(temperature=temperature)

    # LangChain chat model: send system+user if supported, else concatenate
    try:
        from langchain_core.messages import SystemMessage, HumanMessage

        resp = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
        raw = resp.content
    except Exception:
        raw = llm.invoke(f"{sys_prompt}\n\n{user_prompt}").content  # fallback

    try:
        parsed: Dict[str, Any] = _safe_json_load(raw)
    except Exception:
        parsed = {
            "clarification_needed": False,
            "clarification_question": None,
            "intent": "UNKNOWN",
            "rewritten_query": user_query_n,
            "entities": {
                "metric": default_wait_metric,
                "dimension": dimension,
                "timeframe_days": timeframe_days,
                "top_n": top_n,
                "filters": {},
            },
            "assumptions": ["Fallback parse used due to invalid model JSON output."],
        }

    # --- ensure required keys exist ---
    parsed.setdefault("clarification_needed", False)
    parsed.setdefault("clarification_question", None)
    parsed.setdefault("intent", "UNKNOWN")
    parsed.setdefault("rewritten_query", user_query_n)
    parsed.setdefault("entities", {})
    parsed.setdefault("assumptions", [])

    ent = parsed["entities"] or {}
    ent.setdefault("dimension", dimension)
    ent.setdefault("timeframe_days", timeframe_days)
    ent.setdefault("top_n", top_n)
    ent.setdefault("filters", {})

    # =========================
    # ✅ Enterprise "anti-cheap-clarification" layer
    # =========================
    assumptions: List[str] = list(parsed.get("assumptions") or [])

    metric = str(ent.get("metric") or "").strip()
    baseline_metric = str(ent.get("comparison_baseline_metric") or "").strip()

    # Detect when user likely wants comparison ("higher than average")
    # (No hardcoding of a particular airport; purely semantic inference)
    compare_hint = any(
        kw in user_query_n.lower()
        for kw in [
            "higher than average",
            "lower than average",
            "above average",
            "below average",
            "compared to average",
            "vs average",
            "than avg",
        ]
    )
    compare_flag = _boolish(ent.get("compare_to_average")) or _boolish(ent.get("comparison")) or compare_hint

    # If metric is missing/UNKNOWN -> assume sensible default
    if not metric or metric.upper() == "UNKNOWN":
        ent["metric"] = default_wait_metric
        assumptions.append(f"Assumed metric='{default_wait_metric}' from semantic defaults.")

    # If user likely wants comparison but baseline missing -> assume baseline default
    if compare_flag and (not baseline_metric or baseline_metric.upper() == "UNKNOWN"):
        ent["comparison_baseline_metric"] = default_baseline_metric
        assumptions.append(
            f"Assumed comparison baseline='{default_baseline_metric}' for 'average wait time'."
        )
        ent["compare_to_average"] = True  # normalize

    # If model asked for clarification, override it if we can proceed with defaults
    if parsed.get("clarification_needed") is True and prefer_assumptions:
        # "clarify_only_if_unmappable" means: only keep clarification if still missing required fields
        still_unmappable = False
        # minimal check: metric must exist after defaulting
        if not str(ent.get("metric") or "").strip():
            still_unmappable = True

        if clarify_only_if_unmappable and not still_unmappable:
            parsed["clarification_needed"] = False
            parsed["clarification_question"] = None
            assumptions.append("Clarification avoided by applying semantic defaults.")
        elif not clarify_only_if_unmappable:
            parsed["clarification_needed"] = False
            parsed["clarification_question"] = None
            assumptions.append("Clarification avoided by policy (prefer assumptions).")

    # --- enforce policy: at most one clarification question (if any survives) ---
    if max_q <= 1 and parsed.get("clarification_needed") is True:
        q = (parsed.get("clarification_question") or "").strip()
        parsed["clarification_question"] = q.split("\n")[0][:240]

    parsed["entities"] = ent
    parsed["assumptions"] = assumptions

    return parsed