"""Background task queue for long-running browser/code operations."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

TaskStatus = Enum("TaskStatus", "PENDING RUNNING DONE ERROR")


@dataclass
class Task:
    id: str
    name: str
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    done_at: float | None = None


class TaskQueue:
    """Simple in-memory task queue with a background worker thread."""

    def __init__(self):
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

    def submit(self, name: str, fn: Callable[[], Any]) -> str:
        """Queue a task to run in the background. Returns task ID."""
        with self._lock:
            task_id = str(uuid.uuid4())[:8]
            task = Task(id=task_id, name=name)
            self._tasks[task_id] = task

        # Start worker if not running
        if not self._worker or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker.start()

        # Schedule the task for execution
        with self._lock:
            # Queue it by setting status - the worker loop picks up pending tasks
            pass
        # Actually run in background now
        thread = threading.Thread(target=self._run_task, args=(task_id, fn), daemon=True)
        thread.start()

        return task_id

    def _run_task(self, task_id: str, fn: Callable[[], Any]) -> None:
        """Execute a single task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()

        try:
            result = fn()
            with self._lock:
                task = self._tasks.get(task_id)
                if task:
                    task.result = result
                    task.status = TaskStatus.DONE
                    task.done_at = time.time()
        except Exception as e:
            with self._lock:
                task = self._tasks.get(task_id)
                if task:
                    task.error = f"{type(e).__name__}: {e}"
                    task.status = TaskStatus.ERROR
                    task.done_at = time.time()

    def _worker_loop(self) -> None:
        """Background worker that processes queued tasks (currently run inline)."""
        # Tasks are actually executed directly in _run_task spawned as threads
        pass

    def get(self, task_id: str) -> Task | None:
        """Get task by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def list_all(self, limit: int = 50) -> list[dict]:
        """List all tasks, most recent first."""
        with self._lock:
            tasks = sorted(self._tasks.values(), key=lambda t: t.created_at, reverse=True)
            return [
                {
                    "id": t.id,
                    "name": t.name,
                    "status": t.status.name,
                    "result": t.result,
                    "error": t.error,
                    "created_at": t.created_at,
                    "started_at": t.started_at,
                    "done_at": t.done_at,
                }
                for t in tasks[:limit]
            ]


# Global queue instance
_queue: TaskQueue | None = None


def get_queue() -> TaskQueue:
    global _queue
    if _queue is None:
        _queue = TaskQueue()
    return _queue