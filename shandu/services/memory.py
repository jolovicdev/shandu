from __future__ import annotations

from typing import Any

from blackgeorge.memory.base import MemoryStore

from ..contracts import MemoryNote


class MemoryService:
    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    def write(self, scope: str, key: str, value: Any, author: str) -> MemoryNote:
        self._store.write(key, value, scope)
        return MemoryNote(key=key, scope=scope, value=value, author=author)

    def read(self, scope: str, key: str) -> Any | None:
        return self._store.read(key, scope)

    def search(self, scope: str, query: str) -> list[tuple[str, Any]]:
        return self._store.search(query, scope)

    def reset(self, scope: str) -> None:
        self._store.reset(scope)
