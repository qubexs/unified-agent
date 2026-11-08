"""`ua` command line interface."""

from __future__ import annotations

import argparse
import sys

from .agent import Agent, Reply
from .bootstrap import build, doctor
from .config import Settings
from .personas import search_personas


def _approver(allow_all: bool):
    if allow_all:
        return lambda action, task: True

    def ask(action: str, task: str) -> bool:
        print(f"\n  The agent wants to use the [{action}] tool:\n    {task}")
        return input("  Allow? [y/N] ").strip().lower() in {"y", "yes"}

    return ask


def _print_reply(r: Reply) -> None:
    for c in r.tool_calls:
        status = "ran" if c.approved else "declined"
        print(f"  [{c.action} {status}]")
    print(f"\n{r.text}\n")


def _pick_persona(personas, slug: str | None):
    if not slug:
        return None
    if slug not in personas:
        close = [p.slug for p in search_personas(personas, slug)][:5]
        sys.exit(f"Unknown persona '{slug}'." + (f" Did you mean: {', '.join(close)}?" if close else ""))
    return personas[slug]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ua", description="Unified Agent: personas + memory + browser + code.")
    sub = p.add_subparsers(dest="cmd")

    def common(sp):
        sp.add_argument("-p", "--persona", help="persona slug (see `ua personas`)")
        sp.add_argument("--yes", action="store_true", help="auto-approve browser/code tool use")

    chat = sub.add_parser("chat", help="interactive chat (default)")
    common(chat)
    ask = sub.add_parser("ask", help="one-shot question")
    ask.add_argument("text")
    common(ask)

    pl = sub.add_parser("personas", help="list or search personas")
    pl.add_argument("query", nargs="*")

    mem = sub.add_parser("memory", help="search or add long-term memories")
    mem.add_argument("action", choices=["search", "add"])
    mem.add_argument("text", nargs="+")

    sub.add_parser("doctor", help="show what is configured and working")
    sub.add_parser("serve", help="run the local web UI").add_argument("--port", type=int, default=8765)
    sub.add_parser("desktop", help="open the desktop window")

    args = p.parse_args(argv)
    cmd = args.cmd or "chat"
    agent, personas = build()
    agent: Agent

    if cmd == "doctor":
        for name, ok, detail in doctor(agent.settings, agent, personas):
            print(f"{'ok ' if ok else '-- '} {name:<18} {detail}")
        return

    if cmd == "personas":
        items = search_personas(personas, " ".join(args.query)) if args.query else sorted(
            personas.values(), key=lambda x: x.name
        )
        for x in items:
            print(f"{x.slug:<28} {x.description}")
        return

    if cmd == "memory":
        text = " ".join(args.text)
        uid = agent.settings.user_id
        if args.action == "add":
            agent.memory.add(uid, [{"role": "user", "content": text}])
            print("Saved.")
        else:
            for m in agent.memory.search(uid, text, 10) or ["(nothing found)"]:
                print(f"- {m}")
        return

    if cmd == "serve":
        from .server import serve

        serve(agent, personas, host=agent.settings.host, port=args.port)
        return

    if cmd == "desktop":
        from .desktop import run_desktop

        run_desktop(agent, personas)
        return

    persona = _pick_persona(personas, getattr(args, "persona", None))
    approve = _approver(getattr(args, "yes", False))

    if cmd == "ask":
        _print_reply(agent.chat(args.text, persona, approve=approve))
        return

    # interactive chat
    print(f"Unified Agent  (persona: {persona.name if persona else 'Assistant'}, "
          f"tools: {', '.join(agent.available_tools()) or 'none'}). Type /exit to quit.")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text in {"/exit", "/quit"}:
            break
        if text.startswith("/persona "):
            persona = _pick_persona(personas, text.split(None, 1)[1].strip())
            print(f"Persona: {persona.name}")
            continue
        _print_reply(agent.chat(text, persona, approve=approve))


if __name__ == "__main__":
    main()
