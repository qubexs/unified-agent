"""The agent loop: recall memory -> pick an action (structured) -> use tools -> answer -> remember."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Literal

from pydantic import BaseModel, Field

from .config import Settings
from .llm import LLM, Message
from .memory import Memory
from .personas import Persona
from .structured import OutlinesGenerator, generate_structured
from .tools import ToolUnavailable, run_browser_task, run_code_task

DEFAULT_PERSONA = Persona(
    slug="assistant",
    name="Assistant",
    description="General-purpose helpful assistant",
    body="You are a capable, concise assistant.",
)


class Decision(BaseModel):
    """What the agent does next."""

    action: Literal["answer", "browse", "code"] = Field(
        description="answer = reply to the user now; browse = use a web browser; code = delegate a coding task"
    )
    input: str = Field(
        description="For answer: the reply text. For browse/code: a complete, self-contained task description."
    )


@dataclass
class ToolCall:
    action: str
    input: str
    result: str
    approved: bool = True


@dataclass
class Reply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    memories: list[str] = field(default_factory=list)


@dataclass
class RunRecord:
    ts: float
    session: str
    user_text: str
    persona: str
    reply: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    memories: list[str] = field(default_factory=list)


# approve(action, task) -> bool
Approver = Callable[[str, str], bool]


class Agent:
    def __init__(
        self,
        settings: Settings,
        llm: LLM,
        memory: Memory,
        browser_fn: Callable[[str, Settings], str] = run_browser_task,
        code_fn: Callable[[str, Settings], str] = run_code_task,
    ):
        self.settings = settings
        self.llm = llm
        self.memory = memory
        self._browser_fn = browser_fn
        self._code_fn = code_fn
        self._history: dict[str, list[Message]] = {}
        self._runs: list[RunRecord] = []
        self._outlines: OutlinesGenerator | None = None

    # -- helpers -----------------------------------------------------------
    def available_tools(self) -> list[str]:
        tools = []
        if self.settings.browser_enabled:
            tools.append("browse")
        if self.settings.code_cmd.strip():
            tools.append("code")
        return tools

    def get_history(self, session: str = "default") -> list[Message]:
        return list(self._history.get(session, []))

    def get_runs(self, session: str | None = None, limit: int = 20) -> list[dict]:
        runs = [r for r in self._runs if session is None or r.session == session]
        runs = runs[-limit:]
        return [
            {
                "ts": r.ts,
                "session": r.session,
                "user_text": r.user_text,
                "persona": r.persona,
                "reply": r.reply,
                "tool_calls": [
                    {
                        "action": c.action,
                        "input": c.input,
                        "result": c.result,
                        "approved": c.approved,
                    }
                    for c in r.tool_calls
                ],
                "memories": r.memories,
            }
            for r in runs
        ]

    def _system_prompt(self, persona: Persona, memories: list[str]) -> str:
        tool_lines = {
            "browse": "- browse: operate a real web browser to look things up or perform web tasks.",
            "code": "- code: delegate a software task (write/edit/run code) to a coding agent.",
        }
        tools = self.available_tools()
        parts = [
            persona.body,
            "You can take one action per step. Available actions:",
            "- answer: reply to the user (put the full reply in `input`).",
            *[tool_lines[t] for t in tools],
            "Use a tool only when the answer genuinely needs it. Tool results arrive as the next message.",
        ]
        if memories:
            parts.append("What you remember about this user:\n" + "\n".join(f"- {m}" for m in memories))
        return "\n\n".join(parts)

    def _outlines_gen(self) -> OutlinesGenerator | None:
        if not self.settings.outlines_model:
            return None
        if self._outlines is None:
            self._outlines = OutlinesGenerator(self.settings.outlines_model)
        return self._outlines

    # -- main entry point ----------------------------------------------------
    def chat(
        self,
        user_text: str,
        persona: Persona | None = None,
        session: str = "default",
        approve: Approver | None = None,
    ) -> Reply:
        persona = persona or DEFAULT_PERSONA
        uid = self.settings.user_id
        try:
            memories = self.memory.search(uid, user_text, self.settings.memory_top_k)
        except Exception:
            memories = []  # memory must never break a chat
        system = self._system_prompt(persona, memories)

        history = self._history.setdefault(session, [])
        convo: list[Message] = [*history, {"role": "user", "content": user_text}]
        calls: list[ToolCall] = []
        final = ""

        for step in range(self.settings.max_steps):
            try:
                d = generate_structured(
                    self.llm, system, convo, Decision, outlines_gen=self._outlines_gen()
                )
            except ValueError as e:
                # One last fallback: ask the model in plain text to explain
                # what it was trying to do, so the user still gets something.
                # (Happens when the gateway truncates JSON mid-way.)
                try:
                    raw = self.llm.complete(system, convo)
                    final = raw.strip()[:2000] if raw.strip() else f"(I couldn't produce a valid response: {e})"
                    if final.startswith("{"):
                        final = f"(I couldn't produce a valid response: {e})\n\nRaw model output:\n{final}"
                except Exception:
                    final = f"(I couldn't produce a valid response: {e})"
                break

            if d.action == "answer" or d.action not in self.available_tools():
                final = d.input if d.action == "answer" else (
                    f"(The model asked for the '{d.action}' tool, which is not enabled.)"
                )
                break

            # Tool use needs explicit approval: browsing and coding have real side effects.
            if approve is None or not approve(d.action, d.input):
                calls.append(ToolCall(d.action, d.input, "Declined: tool use was not approved.", False))
                convo += [
                    {"role": "assistant", "content": d.model_dump_json()},
                    {"role": "user", "content": "Tool use was not approved. Answer without it."},
                ]
                continue

            try:
                runner = self._browser_fn if d.action == "browse" else self._code_fn
                result = runner(d.input, self.settings)
            except ToolUnavailable as e:
                result = f"Tool unavailable: {e}"
            except Exception as e:  # tools touch the outside world; report, don't crash
                result = f"Tool error: {type(e).__name__}: {e}"
            calls.append(ToolCall(d.action, d.input, result))
            convo += [
                {"role": "assistant", "content": d.model_dump_json()},
                {"role": "user", "content": f"[{d.action} result]\n{result}"},
            ]
        else:
            final = "(Stopped: reached the step limit before finishing.)"

        history += [{"role": "user", "content": user_text}, {"role": "assistant", "content": final}]
        del history[:-20]  # keep the live context bounded; long-term memory holds the rest
        try:
            self.memory.add(uid, [{"role": "user", "content": user_text}, {"role": "assistant", "content": final}])
        except Exception:
            pass
        self._runs.append(
            RunRecord(
                ts=time.time(),
                session=session,
                user_text=user_text,
                persona=persona.slug,
                reply=final,
                tool_calls=calls,
                memories=memories,
            )
        )
        return Reply(text=final, tool_calls=calls, memories=memories)
