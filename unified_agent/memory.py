"""Long-term memory. Uses mem0 when installed and requested, else a local JSON store."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Protocol

from .config import Settings


class Memory(Protocol):
    name: str

    def add(self, user_id: str, messages: list[dict[str, str]]) -> None: ...
    def search(self, user_id: str, query: str, limit: int = 4) -> list[str]: ...
    def export(self, user_id: str) -> list[dict]: ...
    def import_rows(self, rows: list[dict]) -> None: ...


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{3,}", text.lower())}


class LocalMemory:
    """Tiny keyword-overlap memory. Good enough as a fallback; mem0 is much better."""

    name = "local"

    def __init__(self, path: Path):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> list[dict]:
        if not self._path.exists():
            return []
        try:
            return json.loads(self._path.read_text())
        except json.JSONDecodeError:
            return []

    def _save(self, rows: list[dict]) -> None:
        self._path.write_text(json.dumps(rows, indent=2))

    def add(self, user_id: str, messages: list[dict[str, str]]) -> None:
        # Store only what the user said; assistant text is rarely a durable fact.
        rows = self._load()
        for m in messages:
            if m.get("role") == "user" and m.get("content", "").strip():
                rows.append({"user_id": user_id, "text": m["content"].strip(), "ts": time.time()})
        self._save(rows[-2000:])

    def search(self, user_id: str, query: str, limit: int = 4) -> list[str]:
        q = _tokens(query)
        scored = []
        for r in self._load():
            if r["user_id"] != user_id:
                continue
            overlap = len(q & _tokens(r["text"]))
            if overlap:
                scored.append((overlap, r["ts"], r["text"]))
        scored.sort(reverse=True)
        return [text for _, _, text in scored[:limit]]

    def export(self, user_id: str) -> list[dict]:
        return [r for r in self._load() if r.get("user_id") == user_id]

    def import_rows(self, rows: list[dict]) -> None:
        existing = self._load()
        existing.extend(r for r in rows if r.get("text"))
        self._save(existing[-5000:])


class Mem0Memory:
    name = "mem0"

    def __init__(self, memory_obj=None):
        if memory_obj is None:
            from mem0 import Memory as _Mem0

            memory_obj = _Mem0()
        self._m = memory_obj

    def add(self, user_id: str, messages: list[dict[str, str]]) -> None:
        self._m.add(messages, user_id=user_id)

    def search(self, user_id: str, query: str, limit: int = 4) -> list[str]:
        res = self._m.search(query=query, filters={"user_id": user_id}, top_k=limit)
        rows = res.get("results", []) if isinstance(res, dict) else res
        return [r["memory"] for r in rows if r.get("memory")]

    def export(self, user_id: str) -> list[dict]:
        raise NotImplementedError("mem0 does not support bulk export via this wrapper")

    def import_rows(self, rows: list[dict]) -> None:
        raise NotImplementedError("mem0 does not support bulk import via this wrapper")


def make_memory(settings: Settings) -> Memory:
    if settings.memory_backend == "mem0":
        try:
            return Mem0Memory()
        except Exception as e:  # missing package, missing API key, etc.
            print(f"[unified-agent] mem0 unavailable ({e}); using local memory.")
    return LocalMemory(settings.home / "memory.json")
