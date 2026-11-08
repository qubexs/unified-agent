"""Optional cloud sync for settings/memory/presets.

This module pushes/downloads a small JSON snapshot to a user-configured URL.
It is intentionally tiny and dependency-free: it uses urllib from the stdlib.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

from .config import Settings


def sync_enabled(settings: Settings) -> bool:
    return bool(settings.cloud_sync_enabled and settings.cloud_sync_url)


def _read_bundle(settings: Settings) -> dict:
    bundle: dict = {
        "user_id": settings.user_id,
        "settings": settings.to_dict(),
        "snapshot_ts": time.time(),
        "updated_at": time.time(),
    }

    # memory (local JSON store)
    mem_path = settings.home / "memory.json"
    if mem_path.exists():
        try:
            bundle["memory"] = json.loads(mem_path.read_text())
        except Exception:
            pass

    # presets
    presets_path = settings.home / settings.user_id / "presets.json"
    if presets_path.exists():
        try:
            bundle["presets"] = json.loads(presets_path.read_text())
        except Exception:
            pass

    return bundle


def _post(url: str, payload: dict, api_key: str = "") -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=10) as resp:  # nosec: simple local sync
        return json.loads(resp.read().decode("utf-8"))


def _get(url: str, api_key: str = "") -> dict:
    req = urllib.request.Request(url, method="GET")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def push_snapshot(settings: Settings) -> dict:
    if not sync_enabled(settings):
        return {"ok": False, "reason": "sync disabled"}
    bundle = _read_bundle(settings)
    return _post(settings.cloud_sync_url, bundle, settings.cloud_sync_api_key)


def pull_snapshot(settings: Settings, force: bool | None = None) -> dict:
    if not sync_enabled(settings):
        return {"ok": False, "reason": "sync disabled"}
    remote = _get(settings.cloud_sync_url, settings.cloud_sync_api_key)

    # Conflict check: if remote is older than local and not forced, return 409-like dict
    force = force if force is not None else getattr(settings, "sync_force_overwrite", False)
    local_ts_path = settings.home / settings.user_id / "sync_ts.txt"
    local_ts = 0.0
    if local_ts_path.exists():
        try:
            local_ts = float(local_ts_path.read_text())
        except Exception:
            pass
    remote_ts = float(remote.get("snapshot_ts") or remote.get("updated_at") or 0)
    if not force and remote_ts < local_ts:
        return {"ok": False, "conflict": True, "detail": "Remote is older than local", "local_ts": local_ts, "remote_ts": remote_ts}

    # Apply settings if present
    if isinstance(remote.get("settings"), dict):
        from .config import Settings as _S
        new = _S.from_dict(remote["settings"])
        settings.provider = new.provider
        settings.model = new.model
        settings.base_url = new.base_url
        settings.memory_backend = new.memory_backend
        settings.browser_enabled = new.browser_enabled
        settings.code_cmd = new.code_cmd
        settings.save()
    # Apply memory
    if isinstance(remote.get("memory"), list):
        mem_path = settings.home / "memory.json"
        mem_path.parent.mkdir(parents=True, exist_ok=True)
        mem_path.write_text(json.dumps(remote["memory"], indent=2))
    # Apply presets
    if isinstance(remote.get("presets"), list):
        presets_path = settings.home / settings.user_id / "presets.json"
        presets_path.parent.mkdir(parents=True, exist_ok=True)
        presets_path.write_text(json.dumps(remote["presets"], indent=2))

    # Update local timestamp
    local_ts_path.parent.mkdir(parents=True, exist_ok=True)
    local_ts_path.write_text(str(time.time()))

    return {"ok": True, "remote": remote}
