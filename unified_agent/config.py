"""Settings, read from environment variables (UA_* plus the usual API keys)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_PERSONAS = Path(__file__).parent / "personas"
DEFAULT_HOME = Path.home() / ".unified-agent"
DEFAULT_SETTINGS_FILE = "settings.json"


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return value if isinstance(value, Path) else Path(value)


@dataclass
class Settings:
    # LLM
    provider: str = "echo"  # anthropic | openai | echo
    model: str = ""
    base_url: str | None = None  # OpenAI-compatible servers: Ollama, vLLM, ...
    # Storage
    home: Path = field(default_factory=lambda: DEFAULT_HOME)
    user_id: str = "default_user"
    # Memory: "mem0" (falls back to local if unavailable) or "local"
    memory_backend: str = "local"
    memory_top_k: int = 4
    # Personas: extra directories (e.g. a clone of the agency-agents repo)
    persona_dirs: list[Path] = field(default_factory=list)
    # Tools
    browser_enabled: bool = True
    code_cmd: str = ""  # external coding-agent command, task appended as last argument
    code_workdir: Path | None = None
    max_steps: int = 4
    # Outlines (optional, local models only): a Hugging Face model id
    outlines_model: str = ""
    # Local server / Docker
    host: str = "127.0.0.1"
    port: int = 8765
    # Cloud sync (optional)
    cloud_sync_url: str = ""
    cloud_sync_api_key: str = ""
    cloud_sync_enabled: bool = False
    sync_interval: int = 10
    sync_on_startup: bool = False
    sync_on_save: bool = False
    sync_force_overwrite: bool = False

    @property
    def config_path(self) -> Path:
        if self.user_id == "default_user":
            return self.home / DEFAULT_SETTINGS_FILE
        else:
            return self.home / self.user_id / DEFAULT_SETTINGS_FILE

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "home": str(self.home),
            "user_id": self.user_id,
            "memory_backend": self.memory_backend,
            "memory_top_k": self.memory_top_k,
            "persona_dirs": [str(p) for p in self.persona_dirs],
            "browser_enabled": self.browser_enabled,
            "code_cmd": self.code_cmd,
            "code_workdir": str(self.code_workdir) if self.code_workdir else None,
            "max_steps": self.max_steps,
            "outlines_model": self.outlines_model,
            "host": self.host,
            "port": self.port,
            "cloud_sync_url": self.cloud_sync_url,
            "cloud_sync_api_key": self.cloud_sync_api_key,
            "cloud_sync_enabled": self.cloud_sync_enabled,
            "sync_interval": self.sync_interval,
            "sync_on_startup": self.sync_on_startup,
            "sync_on_save": self.sync_on_save,
            "sync_force_overwrite": self.sync_force_overwrite,
        }

    def save(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        return cls(
            provider=data.get("provider", "echo"),
            model=data.get("model", ""),
            base_url=data.get("base_url") or None,
            home=_path(data.get("home")) or DEFAULT_HOME,
            user_id=data.get("user_id", "default_user"),
            memory_backend=data.get("memory_backend", "local"),
            memory_top_k=int(data.get("memory_top_k", 4)),
            persona_dirs=[Path(p) for p in data.get("persona_dirs", []) if p],
            browser_enabled=bool(data.get("browser_enabled", True)),
            code_cmd=data.get("code_cmd", ""),
            code_workdir=_path(data.get("code_workdir")),
            max_steps=int(data.get("max_steps", 4)),
            outlines_model=data.get("outlines_model", ""),
            host=data.get("host", "127.0.0.1"),
            port=int(data.get("port", 8765)),
            cloud_sync_url=data.get("cloud_sync_url", ""),
            cloud_sync_api_key=data.get("cloud_sync_api_key", ""),
            cloud_sync_enabled=bool(data.get("cloud_sync_enabled", False)),
            sync_interval=int(data.get("sync_interval", 10)),
            sync_on_startup=bool(data.get("sync_on_startup", False)),
            sync_on_save=bool(data.get("sync_on_save", False)),
            sync_force_overwrite=bool(data.get("sync_force_overwrite", False)),
        )

    @classmethod
    def from_file(cls, path: Path) -> "Settings":
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text())
        except Exception:
            return cls()
        if not isinstance(raw, dict):
            return cls()
        return cls.from_dict(raw)

    @classmethod
    def load(cls) -> "Settings":
        base = cls.from_file(Path(os.environ.get("UA_SETTINGS_FILE", DEFAULT_HOME / DEFAULT_SETTINGS_FILE)))
        return cls.from_env(base)

    @classmethod
    def from_env(cls, base: "Settings" | None = None) -> "Settings":
        settings = base or cls()
        env = os.environ
        provider = env.get("UA_PROVIDER", "").lower()
        if not provider:
            if env.get("ANTHROPIC_API_KEY"):
                provider = "anthropic"
            elif env.get("OPENAI_API_KEY") or env.get("UA_BASE_URL"):
                provider = "openai"
            else:
                provider = settings.provider
        default_models = {
            "anthropic": "claude-sonnet-5-5",
            "openai": "gpt-5-mini",
            "echo": "echo",
        }
        settings.provider = provider
        settings.model = env.get("UA_MODEL", settings.model or default_models.get(provider, ""))
        settings.base_url = env.get("UA_BASE_URL", settings.base_url or "") or None
        settings.home = Path(env.get("UA_HOME", settings.home))
        settings.user_id = env.get("UA_USER_ID", settings.user_id)
        settings.cloud_sync_url = env.get("UA_CLOUD_SYNC_URL", settings.cloud_sync_url)
        settings.cloud_sync_api_key = env.get("UA_CLOUD_SYNC_KEY", settings.cloud_sync_api_key)
        settings.cloud_sync_enabled = _flag("UA_CLOUD_SYNC", settings.cloud_sync_enabled)
        settings.sync_interval = int(env.get("UA_SYNC_INTERVAL", str(settings.sync_interval)))
        settings.sync_on_startup = _flag("UA_SYNC_ON_STARTUP", settings.sync_on_startup)
        settings.sync_on_save = _flag("UA_SYNC_ON_SAVE", settings.sync_on_save)
        settings.sync_force_overwrite = _flag("UA_SYNC_FORCE_OVERWRITE", settings.sync_force_overwrite)
        settings.memory_backend = env.get("UA_MEMORY", settings.memory_backend).lower()
        settings.memory_top_k = int(env.get("UA_MEMORY_TOP_K", str(settings.memory_top_k)))
        settings.memory_top_k = int(env.get("UA_MEMORY_TOP_K", str(settings.memory_top_k)))
        extra = [Path(p) for p in env.get("UA_PERSONA_DIRS", "").split(os.pathsep) if p]
        settings.persona_dirs = [*settings.persona_dirs, *extra]
        settings.browser_enabled = _flag("UA_BROWSER", settings.browser_enabled)
        settings.code_cmd = env.get("UA_CODE_CMD", settings.code_cmd)
        workdir = env.get("UA_CODE_WORKDIR")
        settings.code_workdir = Path(workdir) if workdir else settings.code_workdir
        settings.max_steps = int(env.get("UA_MAX_STEPS", str(settings.max_steps)))
        settings.outlines_model = env.get("UA_OUTLINES_MODEL", settings.outlines_model)
        settings.host = env.get("UA_HOST", settings.host)
        settings.port = int(env.get("UA_PORT", str(settings.port)))
        return settings

    @property
    def all_persona_dirs(self) -> list[Path]:
        return [PACKAGE_PERSONAS, self.home / "personas", *self.persona_dirs]
