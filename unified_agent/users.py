"""User management: simple username/password store with per-user home directories."""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from pathlib import Path


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    if salt is None:
        salt = secrets.token_urlsafe(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return h.hex(), salt


def verify_password(password: str, hash_hex: str, salt: str) -> bool:
    candidate, _ = hash_password(password, salt)
    return secrets.compare_digest(candidate, hash_hex)


class UserStore:
    """JSON-backed user store."""

    name = "users"

    def __init__(self, path: Path):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text())
        except json.JSONDecodeError:
            return {}

    def _save(self, users: dict) -> None:
        self._path.write_text(json.dumps(users, indent=2))

    def create(self, user_id: str, password: str) -> dict:
        users = self._load()
        if user_id in users:
            raise ValueError("User already exists")
        hash_hex, salt = hash_password(password)
        user = {
            "user_id": user_id,
            "hash": hash_hex,
            "salt": salt,
            "created_at": time.time(),
        }
        users[user_id] = user
        self._save(users)
        return user

    def get(self, user_id: str) -> dict | None:
        return self._load().get(user_id)

    def auth(self, user_id: str, password: str) -> bool:
        user = self.get(user_id)
        if not user:
            return False
        return verify_password(password, user["hash"], user["salt"])

    def list_users(self) -> list[dict]:
        return list(self._load().values())


def make_store(settings: dict) -> UserStore:  # keep generic stub
    # Returns a UserStore bound to settings.home/.ua_users
    return UserStore(Path(settings["home"]) / ".ua_users.json")