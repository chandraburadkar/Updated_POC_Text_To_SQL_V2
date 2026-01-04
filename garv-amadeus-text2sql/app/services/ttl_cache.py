# app/services/ttl_cache.py
from __future__ import annotations

import time
import threading
from typing import Any, Dict, Optional


class TTLCache:
    """
    Simple in-memory TTL cache.
    Thread-safe. Good enough for a single FastAPI worker (demo/prod-lite).
    """

    def __init__(self, ttl_seconds: int = 300, max_items: int = 500):
        self.ttl_seconds = int(ttl_seconds)
        self.max_items = int(max_items)
        self._lock = threading.Lock()
        self._store: Dict[str, Dict[str, Any]] = {}

    def _now(self) -> float:
        return time.time()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            if item["expires_at"] < self._now():
                # expired
                self._store.pop(key, None)
                return None
            return item["value"]

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            # basic eviction if too large (remove oldest)
            if len(self._store) >= self.max_items:
                oldest_key = min(self._store, key=lambda k: self._store[k]["created_at"])
                self._store.pop(oldest_key, None)

            self._store[key] = {
                "value": value,
                "created_at": self._now(),
                "expires_at": self._now() + self.ttl_seconds,
            }

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            now = self._now()
            alive = sum(1 for v in self._store.values() if v["expires_at"] >= now)
            return {
                "ttl_seconds": self.ttl_seconds,
                "max_items": self.max_items,
                "items_total": len(self._store),
                "items_alive": alive,
            }