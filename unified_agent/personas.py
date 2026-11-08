"""Agent personas: markdown files with optional `---` frontmatter (name, description).

Any folder of such files works, including a clone of the agency-agents catalog
(point UA_PERSONA_DIRS at it).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Persona:
    slug: str
    name: str
    description: str
    body: str
    path: Path | None = None


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def parse_persona(text: str, path: Path | None = None) -> Persona | None:
    """Return None when the file has no frontmatter name (e.g. a README)."""
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.DOTALL)
    if not m:
        return None
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line and not line.startswith((" ", "\t")):
            key, _, value = line.partition(":")
            meta[key.strip().lower()] = value.strip().strip("\"'")
    name = meta.get("name")
    if not name:
        return None
    return Persona(
        slug=_slugify(name),
        name=name,
        description=meta.get("description", ""),
        body=m.group(2).strip(),
        path=path,
    )


def load_personas(dirs: list[Path]) -> dict[str, Persona]:
    found: dict[str, Persona] = {}
    for d in dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("*.md")):
            try:
                p = parse_persona(f.read_text(encoding="utf-8"), f)
            except (OSError, UnicodeDecodeError):
                continue
            if p:
                found[p.slug] = p  # later dirs override earlier ones
    return found


def search_personas(personas: dict[str, Persona], query: str) -> list[Persona]:
    words = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if w]
    scored = []
    for p in personas.values():
        hay = f"{p.name} {p.description}".lower()
        score = sum(w in hay for w in words)
        if score:
            scored.append((score, p.name, p))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [p for _, _, p in scored]
