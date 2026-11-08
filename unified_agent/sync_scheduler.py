"""Automatic background sync scheduler."""

from __future__ import annotations

import threading
import time

from .cloud_sync import push_snapshot, sync_enabled
from .config import Settings


class SyncScheduler:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_sync: float | None = None
        self.next_sync: float | None = None

    def start(self) -> None:
        if not sync_enabled(self._settings):
            return
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="ua-sync-scheduler",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            interval = max(1, int(getattr(self._settings, "sync_interval", 10) or 10))
            wait = interval * 60
            self.next_sync = time.time() + wait
            if self._stop.wait(wait):
                break
            try:
                push_snapshot(self._settings)
                self.last_sync = time.time()
            except Exception as e:  # keep scheduler alive on error
                print(f"[ua-sync] background push failed: {e}")
                # try again next cycle

    def status(self) -> dict:
        return {
            "enabled": sync_enabled(self._settings),
            "last_sync": self.last_sync,
            "next_sync": self.next_sync,
            "interval": int(getattr(self._settings, "sync_interval", 10) or 10),
        }


# Global instance
_scheduler: SyncScheduler | None = None


def get_scheduler(settings: Settings) -> SyncScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = SyncScheduler(settings)
        _scheduler.start()
    return _scheduler