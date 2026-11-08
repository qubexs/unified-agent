"""Desktop window: runs the local server in a thread and shows it in a native webview."""

from __future__ import annotations

import socket
import threading
import time
import urllib.request
import webbrowser

from .agent import Agent
from .personas import Persona
from .server import create_app


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_desktop(agent: Agent, personas: dict[str, Persona]) -> None:
    import uvicorn

    port = _free_port()
    config = uvicorn.Config(create_app(agent, personas), host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()

    url = f"http://127.0.0.1:{port}/"
    for _ in range(100):  # wait up to ~10s for the server to come up
        try:
            urllib.request.urlopen(url, timeout=0.5)
            break
        except OSError:
            time.sleep(0.1)

    try:
        import webview  # pywebview
    except ImportError:
        print("pywebview not installed; opening your browser instead "
              "(pip install 'unified-agent[desktop]' for a native window). Ctrl+C to quit.")
        webbrowser.open(url)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        webview.create_window("Unified Agent", url, width=980, height=720)
        webview.start()
    server.should_exit = True
