# app/agents/visualizer.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _is_probably_id(col: str) -> bool:
    c = (col or "").lower()
    return c.endswith("_id") or c in {"id", "uuid", "pk"} or "guid" in c


def _score_datetime_col(name: str) -> int:
    n = (name or "").lower()
    score = 0
    for token in ["date", "time", "ts", "timestamp", "day", "month", "week", "year"]:
        if token in n:
            score += 2
    return score


def _score_metric_col(name: str) -> int:
    n = (name or "").lower()
    score = 0
    for token in ["avg", "mean", "median", "max", "min", "sum", "count", "pct", "percent", "rate", "wait", "delay", "queue", "throughput"]:
        if token in n:
            score += 2
    # de-prioritize ids
    if _is_probably_id(n):
        score -= 5
    return score


def _pick_best(cols: List[str], scorer) -> Optional[str]:
    if not cols:
        return None
    ranked = sorted(cols, key=lambda c: scorer(c), reverse=True)
    return ranked[0] if ranked else None


def _to_datetime_if_possible(df: pd.DataFrame, col: str) -> bool:
    """Try to parse to datetime (non-destructive: writes back only if parse works well)."""
    if col not in df.columns:
        return False
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        return True
    if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
        parsed = pd.to_datetime(df[col], errors="coerce", utc=False, infer_datetime_format=True)
        # accept if a reasonable fraction parsed
        ok_ratio = parsed.notna().mean() if len(parsed) else 0.0
        if ok_ratio >= 0.7:
            df[col] = parsed
            return True
    return False


def suggest_plot(
    df: pd.DataFrame,
    *,
    intent: Optional[str] = None,
    metric_hint: Optional[str] = None,
    dimension_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Suggest a visualization spec. Does NOT plot.

    Returns:
      {
        "type": "bar"|"line"|"table"|None,
        "x": str|None,
        "y": str|None,
        "title": str,
        "reason": str,
        "alternatives": [ {type,x,y,reason} ... ]
      }
    """
    if df is None or df.empty:
        return {
            "type": None,
            "x": None,
            "y": None,
            "title": "No data",
            "reason": "Empty dataframe",
            "alternatives": [],
        }

    cols = list(df.columns)

    # Identify column types
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and not _is_probably_id(c)]
    # candidate datetime cols: actual datetime + best-effort parse for common names
    datetime_cols = [c for c in cols if pd.api.types.is_datetime64_any_dtype(df[c])]
    likely_datetime_names = sorted(cols, key=_score_datetime_col, reverse=True)[:3]
    for c in likely_datetime_names:
        if c not in datetime_cols and _score_datetime_col(c) >= 2:
            if _to_datetime_if_possible(df, c):
                datetime_cols.append(c)

    # categorical: low-cardinality non-numeric
    non_numeric_cols = [c for c in cols if c not in numeric_cols and c not in datetime_cols]
    categorical_cols: List[str] = []
    for c in non_numeric_cols:
        try:
            nunique = df[c].nunique(dropna=True)
            if nunique <= max(30, min(200, int(len(df) * 0.5))):
                if not _is_probably_id(c):
                    categorical_cols.append(c)
        except Exception:
            continue

    # Choose y (metric)
    y = None
    if metric_hint and metric_hint in cols:
        y = metric_hint
    if not y and numeric_cols:
        y = _pick_best(numeric_cols, _score_metric_col) or numeric_cols[0]

    # Choose x
    x_time = None
    if datetime_cols:
        x_time = _pick_best(datetime_cols, _score_datetime_col) or datetime_cols[0]

    x_cat = None
    if dimension_hint and dimension_hint in cols:
        x_cat = dimension_hint
    if not x_cat and categorical_cols:
        x_cat = categorical_cols[0]

    # Decide chart type
    # Heuristic:
    # - TREND or datetime present -> line
    # - RANKING or category+numeric -> bar
    # - otherwise -> table
    intent_norm = (intent or "").upper().strip()

    alternatives: List[Dict[str, Any]] = []

    # Candidate 1: line
    if x_time and y:
        alternatives.append(
            {"type": "line", "x": x_time, "y": y, "reason": "Datetime + numeric mapping"}
        )

    # Candidate 2: bar
    if x_cat and y:
        alternatives.append(
            {"type": "bar", "x": x_cat, "y": y, "reason": "Category + numeric mapping"}
        )

    # Candidate 3: table
    alternatives.append({"type": "table", "x": None, "y": None, "reason": "Fallback view"})

    chosen = None
    if intent_norm == "TREND" and x_time and y:
        chosen = alternatives[0]
    elif intent_norm == "RANKING" and x_cat and y:
        # ensure bar candidate exists
        chosen = next((a for a in alternatives if a["type"] == "bar"), alternatives[0])
    else:
        # prefer line if time exists, else bar, else table
        chosen = next((a for a in alternatives if a["type"] == "line"), None) \
                 or next((a for a in alternatives if a["type"] == "bar"), None) \
                 or next((a for a in alternatives if a["type"] == "table"), None)

    if not chosen:
        chosen = {"type": "table", "x": None, "y": None, "reason": "No suitable chart"}

    title = "Visualization"
    if chosen["type"] == "line":
        title = "Trend over time"
    elif chosen["type"] == "bar":
        title = "Top categories"
    elif chosen["type"] == "table":
        title = "Result table"

    return {
        "type": chosen["type"],
        "x": chosen.get("x"),
        "y": chosen.get("y"),
        "title": title,
        "reason": chosen.get("reason", ""),
        "alternatives": alternatives[:3],
    }