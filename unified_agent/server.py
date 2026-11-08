"""Local web server behind the desktop window (and usable directly in a browser).

Security: binds to 127.0.0.1 only, and every API call must carry a random per-launch
token. Without the token, any web page you visit could talk to localhost and trigger
the browser/code tools.
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .agent import Agent
from .bootstrap import doctor
from .config import Settings
from .personas import Persona
from .presets import PresetsStore
from .cloud_sync import push_snapshot, pull_snapshot, sync_enabled
from .config import Settings
from .presets import PresetsStore
from .sync_scheduler import get_scheduler
from .task_queue import get_queue, TaskStatus
from .users import UserStore
from .tools import ToolUnavailable, run_browser_task, run_code_task

UI_FILE = Path(__file__).parent / "ui.html"


def _build_presets_store(settings: Settings) -> PresetsStore:
    return PresetsStore(settings.home / settings.user_id / "presets.json")


class ChatRequest(BaseModel):
    message: str
    persona: str | None = None
    session: str = "default"
    allow_tools: bool = True


class SettingsPatch(BaseModel):
    provider: str | None = None
    model: str | None = None
    memory_backend: str | None = None
    browser_enabled: bool | None = None
    code_cmd: str | None = None
    outlines_model: str | None = None
    cloud_sync_url: str | None = None
    cloud_sync_api_key: str | None = None
    cloud_sync_enabled: bool | None = None


class BackgroundTaskRequest(BaseModel):
    task: str
    tool: str  # "browse" or "code"


class UserCreateRequest(BaseModel):
    user_id: str
    password: str


class UserLoginRequest(BaseModel):
    user_id: str
    password: str


def create_app(agent: Agent, personas: dict[str, Persona], token: str | None = None) -> FastAPI:
    token = token or secrets.token_urlsafe(24)
    app = FastAPI(title="Unified Agent", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.token = token

    store = UserStore(Path(agent.settings.home) / "users.json")

    def auth(x_ua_token: str = Header(default="")) -> None:
        if not secrets.compare_digest(x_ua_token, token):
            raise HTTPException(status_code=401, detail="bad token")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        html = UI_FILE.read_text(encoding="utf-8").replace("__UA_TOKEN__", token)
        return html

    @app.get("/api/personas", dependencies=[Depends(auth)])
    def list_personas() -> list[dict]:
        return [
            {"slug": p.slug, "name": p.name, "description": p.description}
            for p in sorted(personas.values(), key=lambda x: x.name)
        ]

    @app.get("/api/status", dependencies=[Depends(auth)])
    def status() -> dict:
        return {
            "token": token,
            "settings": {
                "provider": agent.settings.provider,
                "model": agent.settings.model,
                "memory_backend": agent.settings.memory_backend,
                "browser_enabled": agent.settings.browser_enabled,
                "code_cmd": agent.settings.code_cmd,
                "user_id": agent.settings.user_id,
                "host": agent.settings.host,
                "port": agent.settings.port,
            },
            "components": [
                {"component": n, "ok": ok, "detail": d} for n, ok, d in doctor(agent.settings, agent, personas)
            ],
        }

    # --- user management ---
    @app.post("/api/users", dependencies=[Depends(auth)])
    def create_user(req: UserCreateRequest) -> dict:
        try:
            user = store.create(req.user_id, req.password)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))
        return {"ok": True, "user_id": user["user_id"]}

    @app.post("/api/login", dependencies=[Depends(auth)])
    def login(req: UserLoginRequest) -> dict:
        if not store.auth(req.user_id, req.password):
            raise HTTPException(status_code=401, detail="bad credentials")
        new_token = secrets.token_urlsafe(24)
        agent.settings.user_id = req.user_id
        return {"token": new_token, "user_id": req.user_id}

    @app.get("/api/users", dependencies=[Depends(auth)])
    def list_users() -> list[dict]:
        return [{"user_id": u["user_id"], "created_at": u["created_at"]} for u in store.list_users()]

    @app.get("/api/runs", dependencies=[Depends(auth)])
    def runs(session: str = "default") -> list[dict]:
        return agent.get_runs(session)

    @app.post("/api/export/runs", dependencies=[Depends(auth)])
    def export_runs():
        """Export all run data as JSON."""
        runs = agent.get_runs(None, limit=10000)  # all runs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return {
            "filename": f"ua_runs_export_{timestamp}.json",
            "content": json.dumps(runs, indent=2),
            "mime": "application/json",
        }

    @app.post("/api/export/memories", dependencies=[Depends(auth)])
    def export_memories():
        """Export all memory data for the user as JSON."""
        try:
            memories = agent.memory.export(agent.settings.user_id)
        except (NotImplementedError, AttributeError):
            raise HTTPException(status_code=400, detail="Memory backend does not support export")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return {
            "filename": f"ua_memories_export_{timestamp}.json",
            "content": json.dumps(memories, indent=2),
            "mime": "application/json",
        }

    @app.post("/api/import/memories", dependencies=[Depends(auth)])
    async def import_memories(file: UploadFile = File(...)):
        """Import memory data from a JSON file."""
        if not file.filename.endswith('.json'):
            raise HTTPException(status_code=400, detail="Only JSON files are supported")
        
        try:
            content = await file.read()
            data = json.loads(content)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON file")
        
        if not isinstance(data, list):
            raise HTTPException(status_code=400, detail="JSON file must contain an array of memory items")
        
        try:
            agent.memory.import_rows(data)
        except (NotImplementedError, AttributeError):
            raise HTTPException(status_code=400, detail="Memory backend does not support import")
        
        return {"ok": True, "imported": len(data)}

    @app.get("/api/history", dependencies=[Depends(auth)])
    def history(session: str = "default") -> list[dict]:
        return [dict(m) for m in agent.get_history(session)]

    @app.get("/api/memories", dependencies=[Depends(auth)])
    def memories(q: str = "") -> list[str]:
        return agent.memory.search(agent.settings.user_id, q, 20) if q.strip() else []

    @app.get("/api/settings", dependencies=[Depends(auth)])
    def get_settings() -> dict:
        return {
            "config_path": str(agent.settings.config_path),
            "settings": {
                "provider": agent.settings.provider,
                "model": agent.settings.model,
                "base_url": agent.settings.base_url,
                "memory_backend": agent.settings.memory_backend,
                "browser_enabled": agent.settings.browser_enabled,
                "code_cmd": agent.settings.code_cmd,
                "outlines_model": agent.settings.outlines_model,
                "host": agent.settings.host,
                "port": agent.settings.port,
            },
        }

    @app.patch("/api/settings", dependencies=[Depends(auth)])
    def update_settings(patch: SettingsPatch) -> dict:
        for key, value in patch.model_dump(exclude_unset=True).items():
            setattr(agent.settings, key, value)
        agent.settings.save()
        return {
            "ok": True,
            "config_path": str(agent.settings.config_path),
            "settings": {
                "provider": agent.settings.provider,
                "model": agent.settings.model,
                "memory_backend": agent.settings.memory_backend,
                "browser_enabled": agent.settings.browser_enabled,
                "code_cmd": agent.settings.code_cmd,
                "outlines_model": agent.settings.outlines_model,
            },
        }

    @app.post("/api/chat", dependencies=[Depends(auth)])
    def chat(req: ChatRequest) -> dict:
        persona = personas.get(req.persona) if req.persona else None
        if req.persona and persona is None:
            raise HTTPException(status_code=404, detail="unknown persona")
        approve = (lambda action, task: True) if req.allow_tools else None
        r = agent.chat(req.message, persona, req.session, approve=approve)
        return {
            "text": r.text,
            "memories": r.memories,
            "tool_calls": [
                {"action": c.action, "input": c.input, "result": c.result, "approved": c.approved}
                for c in r.tool_calls
            ],
        }

    # --- Background task endpoints ---
    queue = get_queue()

    @app.get("/api/tasks", dependencies=[Depends(auth)])
    def list_tasks(limit: int = 50) -> list[dict]:
        """List all background tasks."""
        return queue.list_all(limit)

    @app.get("/api/tasks/{task_id}", dependencies=[Depends(auth)])
    def get_task(task_id: str) -> dict:
        """Get a specific task by ID."""
        task = queue.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="task not found")
        return {
            "id": task.id,
            "name": task.name,
            "status": task.status.name,
            "result": task.result,
            "error": task.error,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "done_at": task.done_at,
        }

    # --- presets ---
    presets_store = _build_presets_store(agent.settings)

    # (legacy sync status endpoint removed; see /api/sync/status near sync handlers)
    _ = sync_enabled  # keep reference
    # Start background sync scheduler once per app
    try:
        from .sync_scheduler import get_scheduler as _get_sched
        sched = _get_sched(agent.settings)
        sched.start()
    except Exception:
        pass

    @app.post("/api/presets", dependencies=[Depends(auth)])
    def create_preset(payload: dict) -> dict:
        data = {**payload}
        # if the frontend sends current settings in settings field, keep it
        pre = presets_store.create(data)
        return {"ok": True, "id": pre["id"]}

    @app.get("/api/presets", dependencies=[Depends(auth)])
    def list_presets() -> list[dict]:
        return presets_store.list()

    @app.patch("/api/presets/{preset_id}", dependencies=[Depends(auth)])
    def update_preset(preset_id: str, payload: dict) -> dict:
        pre = presets_store.get(preset_id)
        if not pre:
            raise HTTPException(status_code=404, detail="not found")
        # live-update backing store
        pre.update(payload)
        presets = presets_store.list()
        for p in presets:
            if p["id"] == preset_id:
                p.update(payload)
        presets_store._save(presets)  # noqa: access internal for simplicity
        return {"ok": True}

    @app.delete("/api/presets/{preset_id}", dependencies=[Depends(auth)])
    def delete_preset(preset_id: str) -> dict:
        presets_store.remove(preset_id)
        return {"ok": True}

    @app.post("/api/presets/import", dependencies=[Depends(auth)])
    async def import_presets(file: UploadFile = File(...)) -> dict:
        content = await file.read()
        data = json.loads(content)
        count = presets_store.import_many(data)
        return {"imported": count}

    # --- cloud sync (optional) ---
    @app.post("/api/sync/push", dependencies=[Depends(auth)])
    def sync_push() -> dict:
        if not sync_enabled(agent.settings):
            raise HTTPException(status_code=400, detail="cloud sync disabled")
        try:
            result = push_snapshot(agent.settings)
            return {"ok": True, "remote": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/sync/pull", dependencies=[Depends(auth)])
    def sync_pull(force: bool = False) -> dict:
        if not sync_enabled(agent.settings):
            raise HTTPException(status_code=400, detail="cloud sync disabled")
        try:
            result = pull_snapshot(agent.settings, force=force)
            if not result.get("ok") and result.get("conflict"):
                raise HTTPException(status_code=409, detail=result.get("detail") or "conflict")
            return {"ok": True, "remote": result}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/sync/status", dependencies=[Depends(auth)])
    def sync_status() -> dict:
        if not sync_enabled(agent.settings):
            return {"enabled": False, "detail": "cloud sync disabled"}
        sched = get_scheduler(agent.settings)
        sched.start()
        sched.settings = agent.settings
        status = sched.status()
        status["enabled"] = True
        return status

    @app.post("/api/tasks", dependencies=[Depends(auth)])
    def submit_task(req: BackgroundTaskRequest) -> dict:
        """Submit a background task (browse or code)."""
        if req.tool == "browse":
            if not agent.settings.browser_enabled:
                raise HTTPException(status_code=400, detail="browser tool not enabled")
            fn = lambda: run_browser_task(req.task, agent.settings)
            name = f"browse: {req.task[:50]}"
        elif req.tool == "code":
            if not agent.settings.code_cmd.strip():
                raise HTTPException(status_code=400, detail="code tool not configured")
            fn = lambda: run_code_task(req.task, agent.settings)
            name = f"code: {req.task[:50]}"
        else:
            raise HTTPException(status_code=400, detail="invalid tool (use 'browse' or 'code')")

        task_id = queue.submit(name, fn)
        return {"task_id": task_id, "status": "submitted"}

    return app


def serve(agent: Agent, personas: dict[str, Persona], host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    app = create_app(agent, personas)
    print(f"Unified Agent UI: http://{host}:{port}/")
    uvicorn.run(app, host=host, port=port, log_level="warning")