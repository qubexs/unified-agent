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
    """Find the first balanced {...} object in free text (handles code fences/prose)."""
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

    schema_hint = json.dumps(schema.model_json_schema())
    sys_prompt = f"{system}\n\nRespond with ONLY a JSON object matching this schema:\n{schema_hint}"
    convo = list(messages)
    last_err = "no JSON found"
    for _ in range(retries + 1):
        raw = llm.complete(sys_prompt, convo)
        blob = extract_json(raw)
        if blob:
            try:
                return schema.model_validate_json(blob)
            except ValidationError as e:
                last_err = str(e)
        convo = [
            *messages,
            {"role": "assistant", "content": raw},
            {"role": "user", "content": f"That was invalid ({last_err[:300]}). Reply with only the JSON object."},
        ]
    raise ValueError(f"Model did not return valid {schema.__name__}: {last_err[:300]}")
