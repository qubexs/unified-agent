@echo off
setlocal
if "%UA_HOST%"=="" set UA_HOST=127.0.0.1
if "%UA_PORT%"=="" set UA_PORT=8765
rem OmniRoute gateway (from C:\Users\tokki\.config\opencode\opencode.jsonc) — openai-compatible
set OPENAI_API_KEY=sk-4effa4a5a0a15633-dc1977-018e0f83
set UA_PROVIDER=openai
set UA_BASE_URL=http://127.0.0.1:20128/v1
set UA_MODEL=auto/coding
set UA_CODE_CMD=claude -p
python -m unified_agent.cli %*
