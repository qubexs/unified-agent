"""Thin LLM abstraction: Anthropic, OpenAI-compatible (incl. Ollama/vLLM), or a stub."""

from __future__ import annotations

from typing import Protocol

from .config import Settings

Message = dict[str, str]  # {"role": "user"|"assistant", "content": str}


class LLM(Protocol):
    def complete(self, system: str, messages: list[Message]) -> str: ...


class LLMUnavailable(RuntimeError):
    pass


class AnthropicLLM:
    def __init__(self, model: str, max_tokens: int = 4096):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise LLMUnavailable("pip install 'unified-agent[anthropic]'") from e
        self._client = anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens

    def complete(self, system: str, messages: list[Message]) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=messages,
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


class OpenAILLM:
    def __init__(self, model: str, base_url: str | None = None):
        try:
            import openai
        except ImportError as e:  # pragma: no cover
            raise LLMUnavailable("pip install 'unified-agent[openai]'") from e
        # OmniRoute requires a real key; local servers ignore it but SDK insists on one.
        import os
        api_key = os.environ.get("OPENAI_API_KEY") or "not-needed"
        kwargs = {"base_url": base_url, "api_key": api_key} if base_url else {"api_key": api_key}
        self._base_url = base_url
        self._model = model
        self._client = openai.OpenAI(**kwargs)

    def complete(self, system: str, messages: list[Message]) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}, *messages],
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""


class EchoLLM:
    """Used when no API key is configured, so the app still starts and explains itself."""

    def complete(self, system: str, messages: list[Message]) -> str:
        return (
            '{"action": "answer", "input": "No model is configured. Set ANTHROPIC_API_KEY, '
            "OPENAI_API_KEY, or UA_BASE_URL (for a local server) and try again. "
            "Run `ua doctor` to see what is available.\"}"
        )


def make_llm(settings: Settings) -> LLM:
    if settings.provider == "anthropic":
        return AnthropicLLM(settings.model)
    if settings.provider == "openai":
        return OpenAILLM(settings.model, settings.base_url)
    return EchoLLM()
