"""Structured output.

Default: ask the model for JSON, validate with pydantic, retry on failure.
Optional: if UA_OUTLINES_MODEL is set (a local Hugging Face model), Outlines
constrains generation so the JSON is valid by construction.
"""

from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .llm import LLM, Message

T = TypeVar("T", bound=BaseModel)


def extract_json(text: str) -> str | None:
    """Find the first balanced {...} object in free text (handles code fences/prose).

    The downstream schema is the onlyбеж source of truth — we extract any
    balanced {...} that parses as JSON. If the model truncated the output
    (e.g. replied `{"description": "Fetch th...`), no balanced object
    exists and we return None so the caller retries instead of failing
    validation on partial JSON.
    """
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


class OutlinesGenerator:
    """Wraps Outlines for local models. Loaded lazily; heavy dependencies."""

    def __init__(self, model_name: str):
        import outlines
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._model = outlines.from_transformers(
            AutoModelForCausalLM.from_pretrained(model_name, device_map="auto"),
            AutoTokenizer.from_pretrained(model_name),
        )

    def __call__(self, prompt: str, schema: type[T], max_new_tokens: int = 400) -> str:
        return self._model(prompt, schema, max_new_tokens=max_new_tokens)


def generate_structured(
    llm: LLM,
    system: str,
    messages: list[Message],
    schema: type[T],
    retries: int = 2,
    outlines_gen: OutlinesGenerator | None = None,
) -> T:
    if outlines_gen is not None:
        prompt = system + "\n\n" + "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return schema.model_validate_json(outlines_gen(prompt, schema))

    # Minimal human-readable hint — the raw pydantic JSON is what confuses cheap
    # models (they copy it as the output, e.g. {"properties": {...}}).
    props = schema.model_json_schema().get("properties", {})
    hint_parts = []
    for key, spec in props.items():
        enum = spec.get("enum")
        des = spec.get("description", key)
        if enum:
            hint_parts.append(f"  \"{key}\": one of {enum} — {des}")
        else:
            hint_parts.append(f"  \"{key}\": string — {des}")
    schema_hint = "{\n" + ",\n".join(hint_parts) + "\n}"
    sys_prompt = (
        f"{system}\n\n"
        "Respond with ONLY a JSON object (no prose, no markdown, no extra keys) "
        "matching this shape:\n"
        f"{schema_hint}\n"
        "Example: {\"action\": \"browse\", \"input\": \"Visit https://example.com and report the title\"}"
    )
    convo = list(messages)
    last_err = "no JSON found"
    for _ in range(retries + 1):
        raw = llm.complete(sys_prompt, convo)
        blob = extract_json(raw)
        if blob:
            # Unwrap the "properties" echo bug: some models output
            # {"properties": {"action": "browse", ...}} or
            # {"description": "...", "properties": {...}}
            try:
                parsed = json.loads(blob)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict) and "action" not in parsed and isinstance(parsed.get("properties"), dict):
                blob = json.dumps(parsed["properties"])
            if isinstance(parsed, dict) and "action" not in parsed and "input" in parsed:
                blob = json.dumps(parsed)
            try:
                return schema.model_validate_json(blob)
            except ValidationError as e:
                last_err = str(e)
        # If raw was empty or truncated, try a second extraction path: look for any JSON object
        # by brute-force trying json.loads on the whole raw after stripping code fences.
        if not blob:
            # Strip code fences
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned[3:].strip()
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3].strip()
            # Try to parse cleaned as JSON directly
            try:
                obj = json.loads(cleaned)
                # If it's a dict with the expected keys, validate it
                return schema.model_validate(obj)
            except Exception:
                pass
            last_err = "no JSON found; model replied plain text"
        convo = [
            *messages,
            {"role": "assistant", "content": raw},
            {"role": "user", "content": f"That was not a valid Decision JSON ({last_err[:300]}). Reply with ONLY {{\"action\": \"browse\", \"input\": \"...\"}} — no 'properties' wrapper, no description key, no truncation."},
        ]
    raise ValueError(f"Model did not return valid {schema.__name__}: {last_err[:500]}")
