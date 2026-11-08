"""Workspaces (Workplace) and Sessions — proper server-side store.

A Workspace groups Sessions. A Session scopes:
  - conversation history (Agent._history[session])
  - runs (RunRecord.session)
  - segments stored on disk under ~/.unified-agent/<user>/workspaces/...

File layout (per user):
  ~/.unified-agent/<user>/workspaces.json       — { workspaces: {...}, current_workspace, current_session }
  ~/.unified-agent/<user>/sessions/<session>.json — persisted history (bounded, last 20)

Rules:
  - IDs are slugified; name is free text.
  - Never delete the last workspace or last session in a workspace (400).
  - Sessions are created lazily when first used via /api/sessions/ensure or /api/chat.
  - All writes are atomic (tmp + replace).
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path


def _slug(s: str, fallback: str = "default") -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or fallback


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


class WorkspaceStore:
    """Per-user JSON store. Thread-unsafe — call from server thread only (FastAPI sync endpoints)."""

    def __init__(self, user_dir: Path):
        self.user_dir = user_dir
        self.file = user_dir / "workspaces.json"
        self.sessions_dir = user_dir / "sessions"

    # -- load / save -----------------------------------------------------
    def _load(self) -> dict:
        if not self.file.exists():
            return self._default()
        try:
            data = json.loads(self.file.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "workspaces" in data:
                return data
        except Exception:
            pass
        return self._default()

    def _default(self) -> dict:
        now = time.time()
        return {
            "workspaces": {
                "default": {"name": "Default", "sessions": ["default"], "created_at": now, "updated_at": now}
            },
            "current_workspace": "default",
            "current_session": "default",
        }

    def _save(self, data: dict) -> None:
        _atomic_write(self.file, data)

    # -- queries ---------------------------------------------------------
    def list(self) -> dict:
        return self._load()

    def current(self) -> tuple[str, str]:
        d = self._load()
        ws = d.get("current_workspace") or "default"
        if ws not in d["workspaces"]:
            ws = next(iter(d["workspaces"]), "default")
        ss = d.get("current_session") or "default"
        sessions = d["workspaces"][ws]["sessions"]
        if ss not in sessions:
            ss = sessions[0] if sessions else "default"
        return ws, ss

    # -- workspaces ------------------------------------------------------
    def create_workspace(self, name: str) -> dict:
        d = self._load()
        ws_id = _slug(name)
        if ws_id in d["workspaces"]:
            raise ValueError(f"workspace already exists: {ws_id}")
        now = time.time()
        d["workspaces"][ws_id] = {"name": name, "sessions": ["default"], "created_at": now, "updated_at": now}
        d["current_workspace"] = ws_id
        d["current_session"] = "default"
        self._save(d)
        return {"id": ws_id, **d["workspaces"][ws_id]}

    def delete_workspace(self, ws_id: str) -> None:
        d = self._load()
        if ws_id not in d["workspaces"]:
            raise ValueError("workspace not found")
        if len(d["workspaces"]) <= 1:
            raise ValueError("cannot delete the last workspace")
        del d["workspaces"][ws_id]
        if d["current_workspace"] == ws_id:
            d["current_workspace"] = next(iter(d["workspaces"]))
            d["current_session"] = d["workspaces"][d["current_workspace"]]["sessions"][0]
        self._save(d)

    def rename_workspace(self, ws_id: str, name: str) -> dict:
        d = self._load()
        if ws_id not in d["workspaces"]:
            raise ValueError("workspace not found")
        d["workspaces"][ws_id]["name"] = name
        d["workspaces"][ws_id]["updated_at"] = time.time()
        self._save(d)
        return {"id": ws_id, **d["workspaces"][ws_id]}

    # -- sessions --------------------------------------------------------
    def ensure_session(self, session: str, workspace_id: str | None = None) -> dict:
        """Create the session if missing; returns the session record."""
        d = self._load()
        ws_id = workspace_id or d["current_workspace"]
        if ws_id not in d["workspaces"]:
            raise ValueError("workspace not found")
        ss = _slug(session, "default")
        if ss not in d["workspaces"][ws_id]["sessions"]:
            d["workspaces"][ws_id]["sessions"].append(ss)
            d["workspaces"][ws_id]["updated_at"] = time.time()
            d["current_workspace"] = ws_id
            d["current_session"] = ss
            self._save(d)
            self._ensure_session_file(ss)
        else:
            d["current_workspace"] = ws_id
            d["current_session"] = ss
            self._save(d)
            self._ensure_session_file(ss)
        return {"id": ss, "workspace_id": ws_id}

    def list_sessions(self, workspace_id: str | None = None) -> list[dict]:
        d = self._load()
        ws_id = workspace_id or d["current_workspace"]
        if ws_id not in d["workspaces"]:
            raise ValueError("workspace not found")
        out = []
        for ss in d["workspaces"][ws_id]["sessions"]:
            f = self.sessions_dir / f"{ss}.json"
            count = 0
            updated = None
            if f.exists():
                try:
                    rec = json.loads(f.read_text(encoding="utf-8"))
                    count = len(rec.get("messages") or [])
                    updated = rec.get("updated_at")
                except Exception:
                    pass
            out.append({"id": ss, "message_count": count, "updated_at": updated})
        return out

    def rename_session(self, ws_id: str, old: str, new: str) -> dict:
        d = self._load()
        if ws_id not in d["workspaces"]:
            raise ValueError("workspace not found")
        sessions = d["workspaces"][ws_id]["sessions"]
        if old not in sessions:
            raise ValueError("session not found")
        new_s = _slug(new)
        if new_s in sessions and new_s != old:
            raise ValueError("session already exists")
        idx = sessions.index(old)
        sessions[idx] = new_s
        if d["current_session"] == old:
            d["current_session"] = new_s
        self._save(d)
        # rename history file if present
        old_f = self.sessions_dir / f"{old}.json"
        new_f = self.sessions_dir / f"{new_s}.json"
        if old_f.exists() and not new_f.exists():
            os.replace(old_f, new_f)
        return {"id": new_s, "workspace_id": ws_id}

    def delete_session(self, ws_id: str, session: str) -> None:
        d = self._load()
        if ws_id not in d["workspaces"]:
            raise ValueError("workspace not found")
        sessions = d["workspaces"][ws_id]["sessions"]
        if session not in sessions:
            raise ValueError("session not found")
        if len(sessions) <= 1:
            raise ValueError("cannot delete the last session in a workspace")
        sessions.remove(session)
        if d["current_session"] == session:
            d["current_session"] = sessions[0]
        self._save(d)
        f = self.sessions_dir / f"{session}.json"
        if f.exists():
            try:
                f.unlink()
            except Exception:
                pass

    # -- persisted history ------------------------------------------------
    def _ensure_session_file(self, session: str) -> None:
        f = self.sessions_dir / f"{session}.json"
        if not f.exists():
            _atomic_write(f, {"session": session, "workspace_id": None, "messages": [], "updated_at": time.time()})

    def save_history(self, session: str, messages: list[dict]) -> None:
        f = self.sessions_dir / f"{_slug(session)}.json"
        _atomic_write(f, {"session": _slug(session), "messages": messages[-20:], "updated_at": time.time()})

    def load_history(self, session: str) -> list[dict]:
        f = self.sessions_dir / f"{_slug(session)}.json"
        if not f.exists():
            return []
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            return data.get("messages") or []
        except Exception:
            return []
