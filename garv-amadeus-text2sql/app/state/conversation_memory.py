from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str
    ts_utc: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ConversationMemory:
    """
    Simple in-memory conversation store.
    Later we can persist it (Redis / SQLite) for multi-user.
    """
    turns: List[Turn] = field(default_factory=list)

    # last extracted entities we care about for follow-ups
    last_entities: Dict[str, Any] = field(default_factory=dict)

    # last sql (useful for debugging / "show SQL")
    last_sql: str = ""

    def add_user(self, text: str) -> None:
        self.turns.append(Turn(role="user", content=text))

    def add_assistant(self, text: str) -> None:
        self.turns.append(Turn(role="assistant", content=text))

    def set_last_entities(self, entities: Dict[str, Any]) -> None:
        self.last_entities = entities or {}

    def get_context(self, max_turns: int = 6) -> List[Dict[str, str]]:
        """
        Returns last N turns in LangChain-style list format.
        """
        recent = self.turns[-max_turns:]
        return [{"role": t.role, "content": t.content} for t in recent]

    def merge_entities(self, new_entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fill missing entity keys from memory.
        Example: user says "same airport", new_entities has none => use last_entities["airport"].
        """
        merged = dict(self.last_entities or {})
        for k, v in (new_entities or {}).items():
            if v not in (None, "", {}, []):
                merged[k] = v
        return merged