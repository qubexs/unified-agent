"""Builds the agent from settings; shared by the CLI, server and desktop app."""

from __future__ import annotations

import importlib.util

from .agent import Agent
from .config import Settings
from .llm import make_llm
from .memory import make_memory
from .personas import Persona, load_personas


def build(settings: Settings | None = None) -> tuple[Agent, dict[str, Persona]]:
    settings = settings or Settings.load()
    settings.home.mkdir(parents=True, exist_ok=True)
    agent = Agent(settings, make_llm(settings), make_memory(settings))
    return agent, load_personas(settings.all_persona_dirs)


def doctor(settings: Settings, agent: Agent, personas: dict[str, Persona]) -> list[tuple[str, bool, str]]:
    """Returns (component, ok, detail) rows describing what is usable right now."""

    def has(mod: str) -> bool:
        return importlib.util.find_spec(mod) is not None

    return [
        ("model", settings.provider != "echo", f"provider={settings.provider} model={settings.model}"),
        ("memory", True, f"backend={agent.memory.name} (requested: {settings.memory_backend})"),
        ("personas", bool(personas), f"{len(personas)} loaded"),
        ("browser tool", settings.browser_enabled and has("browser_use"),
         "browser-use installed" if has("browser_use") else "pip install 'unified-agent[browser]'"),
        ("coding agent", bool(settings.code_cmd.strip()),
         settings.code_cmd or "set UA_CODE_CMD, e.g. 'claude -p'"),
        ("structured output", True,
         f"outlines ({settings.outlines_model})" if settings.outlines_model else "JSON + pydantic validation"),
        ("desktop window", has("webview"), "pywebview installed" if has("webview") else "pip install 'unified-agent[desktop]' (else opens your browser)"),
    ]
