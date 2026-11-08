#!/usr/bin/env bash
set -euo pipefail
export UA_HOST="${UA_HOST:-127.0.0.1}"
export UA_PORT="${UA_PORT:-8765}"
# OmniRoute gateway (from C:\Users\tokki\.config\opencode\opencode.jsonc) — openai-compatible
export OPENAI_API_KEY="sk-4effa4a5a0a15633-dc1977-018e0f83"
export UA_PROVIDER="openai"
export UA_BASE_URL="http://127.0.0.1:20128/v1"
export UA_MODEL="auto/best-chat"
export UA_BROWSER_MODEL="typhoon/typhoon-v2.5-30b-a3b-instruct"
python -m unified_agent.cli "$@"
