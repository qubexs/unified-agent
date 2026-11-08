# Unified Agent

One app (command line **and** desktop window) that combines ideas from five open-source projects:

| Piece | Project | How it's used here |
|---|---|---|
| Long-term memory | [mem0](https://github.com/mem0ai/mem0) | `UA_MEMORY=mem0` (falls back to a local JSON store) |
| Browser control | [browser-use](https://github.com/browser-use/browser-use) | The `browse` tool |
| Structured output | [Outlines](https://github.com/dottxt-ai/outlines) | Optional, for local models (`UA_OUTLINES_MODEL`) |
| Coding agent | [OpenHands](https://github.com/OpenHands/OpenHands) (or any agent CLI) | The `code` tool runs whatever command you set in `UA_CODE_CMD` |
| Personas | [agency-agents](https://github.com/msitarzewski/agency-agents) catalog | Any folder of markdown files with `name:`/`description:` frontmatter |

The agent loop: recall memories -> pick one action as validated JSON (`answer`, `browse`, `code`) -> run tool (only with your approval) -> answer -> remember.

## Install (single command)

Python 3.11+ required.

### Windows (PowerShell)

```powershell
pip install -e ".[anthropic,openai,memory,browser,desktop]"; npm install -g @anthropic-ai/claude-code; python -m playwright install chromium
```

### macOS / Linux

```bash
pip install -e ".[anthropic,openai,memory,browser,desktop]" && npm install -g @anthropic-ai/claude-code && python -m playwright install chromium
```

- `browser` installs `browser-use` + Playwright (Chromium binary is fetched next).
- `desktop` installs pywebview native window.
- `memory` installs mem0; `anthropic` and `openai` install LLM SDKs.
- `claude-code` is the coding-agent backend (`UA_CODE_CMD=claude -p`).

## Run modes

### Native

```bash
ua doctor
ua
ua serve --port 8765
```

Or on Windows:

```bat
run-native.bat serve
```

Or on macOS/Linux:

```bash
./run-native.sh serve
```

### Docker

```bash
docker compose up --build
```

### Docker with browser profile

```bash
docker compose --profile browser up --build unified-agent-browser
```

## Configure (environment variables)

| Variable | Meaning |
|---|---|
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | Picks the model provider automatically |
| `UA_PROVIDER`, `UA_MODEL` | Override provider (`anthropic`, `openai`) and model name |
| `UA_BASE_URL` | OpenAI-compatible server, e.g. Ollama `http://localhost:11434/v1` |
| `UA_MEMORY` | `local` (default) or `mem0` |
| `UA_PERSONA_DIRS` | Extra persona folders (path-separator separated), e.g. a clone of agency-agents |
| `UA_CODE_CMD` | Coding-agent command; the task is appended as the last argument, e.g. `claude -p` |
| `UA_CODE_WORKDIR` | Directory the coding agent runs in |
| `UA_BROWSER` | `0` to disable the browser tool |
| `UA_OUTLINES_MODEL` | Hugging Face model id; enables Outlines constrained generation |
| `UA_HOME` | Data folder (default `~/.unified-agent`) |
| `UA_HOST` | Server bind host for native mode (default `127.0.0.1`) |
| `UA_PORT` | Server port for native mode (default `8765`) |

## Use

```bash
ua doctor                         # what's configured and working
ua                                # interactive chat
ua ask "summarise X" -p researcher
ua personas coding                # search personas
ua memory add "I prefer short answers"
ua memory search "answers"
ua desktop                        # native window (falls back to your browser)
ua serve --port 8765              # local web UI
```

In chat, `/persona <slug>` switches persona and `/exit` quits.

## Safety model (read this)

- **Tools need approval.** The CLI asks before every browse/code action (`--yes` auto-approves). In the desktop UI, tools are off unless you tick "Allow browser / code tools".
- Web pages the browser agent reads can contain hostile instructions, and results are stored in memory. Don't tick "allow tools" for tasks that touch accounts or money, and don't point the browser at your signed-in profile.
- The `code` tool runs an external command with no shell. Point it at a sandboxed agent (e.g. OpenHands in Docker) when possible.
- The local server binds to `127.0.0.1` by default and requires a random per-launch token, so other web pages can't drive it.

## Notes and limits

- Tested here with a scripted model and stubs (see `tests/`). Not tested against live Anthropic/OpenAI, real mem0, real browser-use, pywebview windows, or OpenHands: those need your keys and desktop.
- The `mem0` and `browser-use` calls follow their public READMEs; if a newer release changes an API, the fix is local to `memory.py` / `tools.py`.
- `UA_CODE_CMD` is a generic hook. I did not verify the OpenHands CLI flags, so check its docs for the right command.
- Persona files are read as plain markdown with simple `key: value` frontmatter (no YAML library).

## Package as a standalone app

```bash
pip install pyinstaller
pyinstaller --noconfirm --windowed --name UnifiedAgent --collect-data unified_agent -c unified_agent/cli.py
```

(Untested; PyInstaller often needs extra flags for pywebview and browser-use.)

## Tests

```bash
pip install -e ".[dev]" && pytest
```
