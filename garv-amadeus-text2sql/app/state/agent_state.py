# app/state/agent_state.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """
    Shared state object passed through the Text2SQL pipeline.

    Notebook-friendly:
    - candidate_sql can be dict (metadata) or string
    - dataframe stores executor output dict (with df + preview)
    - result_df stores direct pandas df reference for convenience
    - chat_history + memory_entities enable multi-turn follow-ups
    """

    # -----------------------------
    # Inputs
    # -----------------------------
    user_question: str

    # -----------------------------
    # Multi-turn memory (NEW)
    # -----------------------------
    chat_history: List[Dict[str, str]] = Field(default_factory=list)
    memory_entities: Dict[str, Any] = Field(default_factory=dict)

    # -----------------------------
    # Step 5: Query Rewriter
    # -----------------------------
    rewritten_query: str = ""
    intent: str = "UNKNOWN"
    entities: Dict[str, Any] = Field(default_factory=dict)

    # -----------------------------
    # Step 4: Schema RAG
    # -----------------------------
    schema_context: str = ""
    retrieved_tables: List[str] = Field(default_factory=list)

    # -----------------------------
    # Step 6: SQL Generator
    # -----------------------------
    candidate_sql: Any = None  # dict returned by generate_sql() or raw string

    # -----------------------------
    # Step 7: SQL Validator + Auto-fix
    # -----------------------------
    validation_ok: bool = False
    final_sql: str = ""
    fixed_by_llm: bool = False

    # -----------------------------
    # Step 8: SQL Execution
    # executor returns dict: {row_count, columns, df, preview_markdown}
    # -----------------------------
    dataframe: Optional[Dict[str, Any]] = None
    result_df: Optional[Any] = None  # pandas.DataFrame stored here (optional shortcut)

    # -----------------------------
    # Step 9: Explanation
    # -----------------------------
    explanation: Optional[Dict[str, Any]] = None

    # -----------------------------
    # Optional: Visualization
    # -----------------------------
    chart_path: Optional[str] = None

    # -----------------------------
    # Debug / tracing info
    # -----------------------------
    debug: Dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "arbitrary_types_allowed": True,  # allows pandas DF in result_df/dataframe["df"]
        "extra": "allow",                 # safe if you add new fields later
    }