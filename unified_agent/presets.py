"""Settings presets: named snapshots of Settings that can be stored per user."""

from __future__ import annotations

import json
import time
import uuid

from .config import Settings


class PresetsStore:
    def __init__(self, path):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> list[dict]:
        if not self._path.exists():
            return []
        try:
            return json.loads(self._path.read_text())
        except json.JSONDecodeError:
            return []

    def _save(self, presets: list[dict]) -> None:
        self._path.write_text(json.dumps(presets, indent=2))

    def list(self) -> list[dict]:
        return [p for p in self._load()]

    def get(self, preset_id):
        for p in self._load():
            if p["id"] == preset_id:
                return p
        return None

    def create(self, data: dict) -> dict:
        presets = self._load()
        preset = {
            "id": uuid.uuid4().hex[:8],
            "name": data.get("name", "Untitled"),
            "description": data.get("description", ""),
            "settings": data.get("settings", {}),
            "created_at": time.time(),
        }
        presets.append(preset)
        self._save(presets)
        return preset

    def remove(self, preset_id: str) -> None:
        presets = [p for p in self._load() if p["id"] != preset_id]
        self._save(presets)

    def import_many(self, presets: list[dict]) -> int:
        existing = self._load()
        for p in presets:
            preset = {
                "id": uuid.uuid4().hex[:8],
                "name": p.get("name", "Imported"),
                "description": p.get("description", ""),
                "settings": p.get("settings", {}),
                "created_at": time.time(),
                "imported_at": time.time(),
            }
            existing.append(preset)
        self._save(existing)
        return len(presets)
