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

    # Use separate browser model if configured (auto/* routers are non-deterministic and
    # cause sporadic 404 model_not_found on structured steps like step 2+)
    model = (getattr(settings, "browser_model", "") or settings.model)

    if settings.provider == "anthropic":
        from browser_use import ChatAnthropic  # type: ignore

        llm = ChatAnthropic(model=model)
    elif settings.provider == "openai":
        from browser_use import ChatOpenAI  # type: ignore

        # OmniRoute (and other OpenAI-compatible gateways) may not support strict
        # structured output; disable it so browser-use falls back to plain JSON.
        # See: ChatOpenAI(..., dont_force_structured_output=True, add_schema_to_system_prompt=True)
        kwargs = {
            "model": model,
            "dont_force_structured_output": True,
            "add_schema_to_system_prompt": True,
        }
        if settings.base_url:
            kwargs["base_url"] = settings.base_url
        import os

        api_key = os.environ.get("OPENAI_API_KEY") or "not-needed"
        if settings.base_url:
            kwargs["api_key"] = api_key
        try:
            llm = ChatOpenAI(**kwargs)
        except TypeError:
            try:
                llm = ChatOpenAI(model=model)
            except Exception as e2:
                raise ToolUnavailable(f"ChatOpenAI init failed: {e2}") from e2
    else:
        raise ToolUnavailable("Browser tool needs an Anthropic or OpenAI model configured.")

    async def _run() -> str:
        browser = Browser()  # BrowserSession in newer browser-use
        agent = Agent(task=task, llm=llm, browser=browser, max_failures=3)
        history = await agent.run()
        # On browser-use failures (e.g., gateway 404 model_not_found), include history for debugging
        res = history.final_result()
        if res:
            return res
        # Summarize last error from history for visibility instead of empty result
        try:
            if history.history:
                last = history.history[-1]
                if last.result:
                    for r in last.result:
                        if getattr(r, "error", None):
                            return f"(browser agent error: {r.error})"
        except Exception:
            pass
        return "(browser agent finished without a result)"

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
