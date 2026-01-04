# app/agents/followup_suggester.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pandas as pd

from app.agents.llm_factory import get_llm
from app.utils.config_loader import load_yaml_config


def _safe_json_extract(raw: str) -> Dict[str, Any]:
    """
    Extract the first JSON object from raw LLM output safely.
    Returns {} if parsing fails.
    """
    s = (raw or "").strip()
    if not s:
        return {}

    # Attempt to slice to the first JSON object
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        s = s[start : end + 1]

    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _coerce_suggestions(obj: Dict[str, Any], max_suggestions: int) -> List[str]:
    """
    Accept multiple possible key names because LLMs sometimes drift.
    """
    # Preferred key
    candidates = obj.get("suggested_questions")

    # Fallback keys
    if candidates is None:
        candidates = obj.get("suggestions")
    if candidates is None:
        candidates = obj.get("followups")
    if candidates is None:
        candidates = obj.get("next_questions")
    if candidates is None:
        candidates = obj.get("questions")

    if not isinstance(candidates, list):
        return []

    clean: List[str] = []
    for s in candidates:
        q = str(s).strip()
        if not q:
            continue
        # De-dupe ignoring case
        if q.lower() in [x.lower() for x in clean]:
            continue
        # Keep short like chips
        if len(q.split()) > 20:
            q = " ".join(q.split()[:20])
        clean.append(q)

    return clean[:max_suggestions]


def _df_profile(df: Optional[pd.DataFrame], max_cols: int = 12) -> Dict[str, Any]:
    """
    Tiny dataframe profile to ground followup generation.
    Never send full rows (too big).
    """
    if df is None or df.empty:
        return {"has_data": False}

    cols = list(df.columns)[:max_cols]
    d: Dict[str, Any] = {"has_data": True, "columns": cols, "row_count": int(df.shape[0])}

    # Add min/max for numeric columns (small grounding)
    try:
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        stats = {}
        for c in numeric_cols[:6]:
            series = df[c].dropna()
            if series.empty:
                continue
            stats[c] = {"min": float(series.min()), "max": float(series.max())}
        if stats:
            d["numeric_ranges"] = stats
    except Exception:
        pass

    return d


def suggest_followup_questions(
    *,
    user_question: str,
    rewritten_query: str,
    intent: str,
    entities: Dict[str, Any],
    retrieved_tables: List[str],
    final_sql: str,
    schema_context: str,
    df: Optional[pd.DataFrame] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.2,
    max_suggestions: int = 3,
) -> List[str]:
    """
    Returns a list of follow-up questions.
    Robust to JSON drift and model formatting differences.
    """

    # Optional semantic layer grounding
    try:
        semantic = load_yaml_config("config/amadeus_semantic_layer.yaml")
    except Exception:
        semantic = {}

    df_info = _df_profile(df)

    # Keep small chat context
    turns = (chat_history or [])[-6:]
    history_str = "\n".join(
        [f"{t.get('role')}: {t.get('content')}" for t in turns if t.get("role") and t.get("content")]
    ) or "None"

    prompt = f"""
You are an enterprise Airport Ops analytics copilot.

Task:
Generate {max_suggestions} smart, non-repetitive follow-up questions the user is likely to ask next.

Rules:
- Must relate to the user's last question and the result.
- Must be valid for airport ops analytics.
- Do NOT ask generic clarifying questions like "Which airport?" unless absolutely necessary.
- Prefer actionable followups: breakdowns, trends, drivers, comparisons, anomalies, and recommendations.
- Each suggestion <= 14 words.
- Return JSON ONLY in EXACT format:
  {{
    "suggested_questions": ["...","...","..."]
  }}

Context:
- Intent: {intent}
- Entities: {json.dumps(entities or {}, ensure_ascii=False)}
- Retrieved tables: {json.dumps(retrieved_tables or [], ensure_ascii=False)}
- Final SQL: {final_sql}
- Data summary: {json.dumps(df_info, ensure_ascii=False)}
- Recent conversation: {history_str}
- Semantic layer (may help): {json.dumps((semantic.get("metrics") or {}), ensure_ascii=False)[:2500]}
User asked:
{user_question}

Rewritten query:
{rewritten_query}
""".strip()

    llm = get_llm(temperature=temperature)

    # Invoke
    try:
        from langchain_core.messages import HumanMessage

        raw = llm.invoke([HumanMessage(content=prompt)]).content
    except Exception:
        raw = llm.invoke(prompt).content

    obj = _safe_json_extract(raw)
    sugg = _coerce_suggestions(obj, max_suggestions=max_suggestions)

    return sugg