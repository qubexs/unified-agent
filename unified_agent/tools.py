"""External tools: a browser agent (browser-use) and a pluggable coding agent."""

from __future__ import annotations

import asyncio
import shlex
import subprocess

from .config import Settings


class ToolUnavailable(RuntimeError):
    pass


def run_browser_task(task: str, settings: Settings) -> str:
    """Drive a real browser with browser-use. Sync wrapper (call from a worker thread)."""
    try:
        from browser_use import Agent, Browser  # type: ignore
    except ImportError as e:
        raise ToolUnavailable("Browser tool needs: pip install 'unified-agent[browser]'") from e

    if settings.provider == "anthropic":
        from browser_use import ChatAnthropic  # type: ignore

        llm = ChatAnthropic(model=settings.model)
    elif settings.provider == "openai":
        from browser_use import ChatOpenAI  # type: ignore

        llm = ChatOpenAI(model=settings.model)
    else:
        raise ToolUnavailable("Browser tool needs an Anthropic or OpenAI model configured.")

    async def _run() -> str:
        agent = Agent(task=task, llm=llm, browser=Browser())
        history = await agent.run()
        return history.final_result() or "(browser agent finished without a result)"

    return asyncio.run(_run())


def run_code_task(task: str, settings: Settings) -> str:
    """Hand a coding task to an external agent CLI configured in UA_CODE_CMD.

    The command is split with shlex and the task is appended as the final argument,
    e.g. UA_CODE_CMD="claude -p". No shell is involved.
    """
    if not settings.code_cmd.strip():
        raise ToolUnavailable("No coding agent configured. Set UA_CODE_CMD (e.g. 'claude -p').")
    argv = [*shlex.split(settings.code_cmd), task]
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=900,
            cwd=settings.code_workdir,
        )
    except FileNotFoundError as e:
        raise ToolUnavailable(f"Coding agent command not found: {argv[0]}") from e
    except subprocess.TimeoutExpired:
        return "(coding agent timed out after 15 minutes)"
    out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.returncode else "")
    return out.strip()[-6000:] or f"(coding agent exited with code {proc.returncode}, no output)"
