import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from unified_agent.agent import Agent, Decision
from unified_agent.config import Settings
from unified_agent.memory import LocalMemory, Mem0Memory
from unified_agent.personas import load_personas, parse_persona, search_personas
from unified_agent.server import create_app
from unified_agent.structured import extract_json, generate_structured
from unified_agent.tools import ToolUnavailable, run_code_task


class ScriptedLLM:
    """Returns queued replies; records the system prompts it was given."""

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.systems: list[str] = []

    def complete(self, system, messages):
        self.systems.append(system)
        return self.replies.pop(0)


def decision(action, text):
    return json.dumps({"action": action, "input": text})


def make_agent(tmp_path, llm, **overrides):
    s = Settings(home=tmp_path, **overrides)
    return Agent(
        s,
        llm,
        LocalMemory(tmp_path / "mem.json"),
        browser_fn=lambda task, st: f"page says: {task}",
        code_fn=lambda task, st: f"coded: {task}",
    )


# --- personas ---------------------------------------------------------------
def test_parse_persona_and_skip_readme():
    p = parse_persona("---\nname: Data Nerd\ndescription: \"crunches numbers\"\n---\nBody here")
    assert p and p.slug == "data-nerd" and p.description == "crunches numbers" and p.body == "Body here"
    assert parse_persona("# Just a README") is None
    assert parse_persona("---\ndescription: no name\n---\nx") is None


def test_builtin_personas_load_and_search():
    personas = load_personas(Settings().all_persona_dirs)
    assert {"researcher", "software-engineer", "product-manager", "writer"} <= set(personas)
    assert search_personas(personas, "debug code")[0].slug == "software-engineer"


def test_later_dirs_override(tmp_path):
    (tmp_path / "writer.md").write_text("---\nname: Writer\ndescription: custom\n---\nmine")
    personas = load_personas([*Settings().all_persona_dirs, tmp_path])
    assert personas["writer"].body == "mine"


# --- memory -------------------------------------------------------------------
def test_local_memory_roundtrip_and_user_isolation(tmp_path):
    m = LocalMemory(tmp_path / "m.json")
    m.add("alice", [{"role": "user", "content": "I prefer vim keybindings"},
                    {"role": "assistant", "content": "ignored"}])
    m.add("bob", [{"role": "user", "content": "I prefer emacs keybindings"}])
    assert m.search("alice", "which keybindings do I like") == ["I prefer vim keybindings"]
    assert m.search("alice", "unrelated astronomy") == []


def test_mem0_wrapper_uses_documented_call_shapes():
    calls = {}

    class Fake:
        def add(self, messages, user_id):
            calls["add"] = (messages, user_id)

        def search(self, query, filters, top_k):
            calls["search"] = (query, filters, top_k)
            return {"results": [{"memory": "likes tea"}, {"memory": ""}]}

    m = Mem0Memory(Fake())
    m.add("u1", [{"role": "user", "content": "hi"}])
    assert m.search("u1", "drink?", 3) == ["likes tea"]
    assert calls["add"][1] == "u1" and calls["search"][1] == {"user_id": "u1"} and calls["search"][2] == 3


# --- structured output -------------------------------------------------------
def test_extract_json_handles_fences_and_braces_in_strings():
    text = 'Sure!\n```json\n{"action": "answer", "input": "use {braces} \\" ok"}\n```'
    assert json.loads(extract_json(text))["input"] == 'use {braces} " ok'
    assert extract_json("no json here") is None


def test_generate_structured_retries_then_succeeds():
    llm = ScriptedLLM("not json", '{"action": "bogus", "input": "x"}', decision("answer", "fine"))
    d = generate_structured(llm, "sys", [{"role": "user", "content": "hi"}], Decision)
    assert d.action == "answer" and d.input == "fine"


def test_generate_structured_gives_up():
    with pytest.raises(ValueError):
        generate_structured(ScriptedLLM("x", "y", "z"), "s", [{"role": "user", "content": "hi"}], Decision)


# --- agent --------------------------------------------------------------------
def test_agent_answers_and_remembers(tmp_path):
    llm = ScriptedLLM(decision("answer", "Hello!"), decision("answer", "You like vim."))
    agent = make_agent(tmp_path, llm)
    assert agent.chat("I like vim keybindings").text == "Hello!"
    r = agent.chat("what keybindings do I like?")
    assert r.memories == ["I like vim keybindings"]
    assert "I like vim keybindings" in llm.systems[1]


def test_persona_body_reaches_system_prompt(tmp_path):
    llm = ScriptedLLM(decision("answer", "ok"))
    agent = make_agent(tmp_path, llm)
    persona = load_personas(Settings().all_persona_dirs)["researcher"]
    agent.chat("hi", persona)
    assert "research analyst" in llm.systems[0]


def test_tool_requires_approval(tmp_path):
    llm = ScriptedLLM(decision("browse", "find X"), decision("answer", "Could not browse."))
    agent = make_agent(tmp_path, llm)
    r = agent.chat("look up X")  # no approver -> declined
    assert r.text == "Could not browse."
    assert r.tool_calls[0].approved is False and "Declined" in r.tool_calls[0].result


def test_tool_runs_when_approved_and_result_feeds_back(tmp_path):
    llm = ScriptedLLM(decision("browse", "find X"), decision("answer", "X is great"))
    agent = make_agent(tmp_path, llm)
    r = agent.chat("look up X", approve=lambda a, t: True)
    assert r.text == "X is great"
    assert r.tool_calls[0].result == "page says: find X"


def test_disabled_tool_is_not_run(tmp_path):
    llm = ScriptedLLM(decision("code", "rm -rf"))
    agent = make_agent(tmp_path, llm)  # code_cmd empty -> code tool disabled
    r = agent.chat("do it", approve=lambda a, t: True)
    assert "not enabled" in r.text and not r.tool_calls


def test_tool_errors_are_reported_not_raised(tmp_path):
    llm = ScriptedLLM(decision("browse", "x"), decision("answer", "sorry"))
    agent = make_agent(tmp_path, llm)
    agent._browser_fn = lambda t, s: (_ for _ in ()).throw(ToolUnavailable("not installed"))
    r = agent.chat("x", approve=lambda a, t: True)
    assert "Tool unavailable: not installed" in r.tool_calls[0].result and r.text == "sorry"


def test_step_limit(tmp_path):
    llm = ScriptedLLM(*[decision("browse", "again")] * 3)
    agent = make_agent(tmp_path, llm, max_steps=3)
    assert "step limit" in agent.chat("loop", approve=lambda a, t: True).text


def test_agent_records_runs(tmp_path):
    llm = ScriptedLLM(decision("answer", "hi"))
    agent = make_agent(tmp_path, llm)
    agent.chat("hello there", session="s1")
    runs = agent.get_runs("s1")
    assert runs and runs[-1]["user_text"] == "hello there" and runs[-1]["reply"] == "hi"


def test_settings_save_load_roundtrip(tmp_path):
    settings = Settings(home=tmp_path, provider="openai", model="gpt-5-mini", memory_backend="mem0", browser_enabled=False, code_cmd="claude -p")
    settings.save()
    loaded = Settings.from_file(settings.config_path)
    assert loaded.provider == "openai"
    assert loaded.model == "gpt-5-mini"
    assert loaded.memory_backend == "mem0"
    assert loaded.browser_enabled is False
    assert loaded.code_cmd == "claude -p"


from unified_agent.task_queue import TaskQueue, TaskStatus

# --- code tool ----------------------------------------------------------------
def test_code_tool_runs_command_without_shell(tmp_path):
    s = Settings(home=tmp_path, code_cmd="echo")
    assert run_code_task("hello; echo injected", s) == "hello; echo injected"


def test_code_tool_unconfigured_and_missing():
    with pytest.raises(ToolUnavailable):
        run_code_task("x", Settings(code_cmd=""))
    with pytest.raises(ToolUnavailable):
        run_code_task("x", Settings(code_cmd="definitely-not-a-real-binary-xyz"))


# --- server -------------------------------------------------------------------
def test_server_requires_token_and_chats(tmp_path):
    llm = ScriptedLLM(decision("answer", "hi from server"))
    agent = make_agent(tmp_path, llm)
    personas = load_personas(Settings().all_persona_dirs)
    client = TestClient(create_app(agent, personas, token="secret"))

    assert client.get("/api/personas").status_code == 401
    assert client.get("/api/personas", headers={"X-UA-Token": "wrong"}).status_code == 401
    h = {"X-UA-Token": "secret"}
    assert len(client.get("/api/personas", headers=h).json()) >= 4
    assert 'const TOKEN = "secret"' in client.get("/").text
    r = client.post("/api/chat", headers=h, json={"message": "hello", "persona": "writer"})
    assert r.json()["text"] == "hi from server"
    assert client.post("/api/chat", headers=h, json={"message": "x", "persona": "nope"}).status_code == 404
    assert client.get("/api/runs", headers=h).status_code == 200
    assert client.patch("/api/settings", headers=h, json={"browser_enabled": False}).json()["ok"] is True
    assert client.get("/api/status", headers=h).json()["settings"]["provider"] == "echo"
    settings = client.patch("/api/settings", headers=h, json={"browser_enabled": False}).json()
    assert settings["ok"] is True and Path(settings["config_path"]).name == "settings.json"


def test_server_returns_saved_settings_path(tmp_path):
    llm = ScriptedLLM(decision("answer", "ok"))
    agent = make_agent(tmp_path, llm)
    personas = load_personas(Settings().all_persona_dirs)
    client = TestClient(create_app(agent, personas, token="secret"))
    h = {"X-UA-Token": "secret"}
    data = client.get("/api/settings", headers=h).json()
    assert "config_path" in data and data["settings"]["provider"] == "echo"

def test_background_task_queue():
    from unified_agent.task_queue import TaskQueue, TaskStatus

    tq = TaskQueue()

    def sample_job():
        return "task completed"

    task_id = tq.submit("sample task", sample_job)

    # Make sure the job runs and completes
    import time
    time.sleep(0.1)  # give some time to background processing

    task = tq.get(task_id)
    assert task is not None
    assert task.status == TaskStatus.DONE
    assert task.result == "task completed"
    assert task.error is None

    # Verifying the task list
    tasks = tq.list_all()
    assert len(tasks) == 1
    assert tasks[0]["id"] == task_id
    assert tasks[0]["status"] == "DONE"
