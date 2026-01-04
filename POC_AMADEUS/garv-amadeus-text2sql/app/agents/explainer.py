# app/agents/explainer.py

from __future__ import annotations

from app.audit.langsmith_tracing import traceable_fn
from typing import Any, Dict, Optional
import pandas as pd

from app.agents.llm_factory import get_llm

@traceable_fn("answer_explainer")
def explain_answer(
    user_question: str,
    sql: str,
    df: Any,  # can be DataFrame OR your execute_sql output dict
    intent: str = "UNKNOWN",                       # <-- added
    entities: Optional[Dict[str, Any]] = None,     # <-- added
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Explain results in plain English for business users.

    Accepts:
      - df as a pandas DataFrame OR as execute_sql() output dict {df, preview_markdown, ...}
    """

    entities = entities or {}

    # If df is the executor output dict, extract the DataFrame
    if isinstance(df, dict) and "df" in df:
        df_obj = df["df"]
    else:
        df_obj = df

    if not isinstance(df_obj, pd.DataFrame):
        return {
            "summary": "I could not generate an explanation because result is not a DataFrame.",
            "bullets": [],
        }

    # Use a small preview to avoid sending huge tables to LLM
    preview_md = df_obj.head(20).to_markdown(index=False) if not df_obj.empty else "No rows returned."

    llm = get_llm(temperature=temperature)

    prompt = f"""
You are an airport operations analytics assistant.
Explain the answer clearly to a business user.

User Question:
{user_question}

Intent:
{intent}

Entities (JSON):
{entities}

SQL executed:
{sql}

Result preview (first rows):
{preview_md}

Return:
1) A short 2-3 line summary
2) 3-5 bullet points with key insights
3) Any assumptions or caveats (if needed)
"""

    try:
        resp = llm.invoke(prompt)
        text = getattr(resp, "content", str(resp))

        return {
            "summary": text.strip(),
            "bullets": [],
        }
    except Exception as e:
        return {
            "summary": f"Explanation failed: {e}",
            "bullets": [],
        }