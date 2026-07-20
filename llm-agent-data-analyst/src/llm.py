"""LLM client abstraction.

The agent talks to an `LLM` with a single `create()` method returning an
`LLMResponse` (a stop reason plus Anthropic-format content blocks). Two
implementations:

- AnthropicLLM : real calls to Claude via the `anthropic` SDK. The import
  is lazy so this module (and the agent, and the mock tests) load without
  the SDK installed.
- MockLLM      : returns a scripted sequence of LLMResponses. Lets the
  full agent loop -- tool_use handling, multi-step orchestration, answer
  extraction -- be tested deterministically with no API key or network.

Content blocks are plain dicts in Anthropic's shape, so they can be
appended straight into the message history either way.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class LLMResponse:
    stop_reason: str
    content: list[dict]        # Anthropic-format blocks: text / tool_use

    def tool_uses(self) -> list[dict]:
        return [b for b in self.content if b.get("type") == "tool_use"]

    def text(self) -> str:
        return "".join(b.get("text", "") for b in self.content
                       if b.get("type") == "text")


class MockLLM:
    """Returns pre-scripted responses in order -- for tests and offline demos."""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, system: str, messages: list[dict], tools: list[dict]):
        self.calls.append({"system": system, "messages": messages, "tools": tools})
        if not self._responses:
            raise AssertionError("MockLLM ran out of scripted responses")
        return self._responses.pop(0)


class AnthropicLLM:
    """Real Claude client. Requires `anthropic` and ANTHROPIC_API_KEY."""

    def __init__(self, model: str | None = None, max_tokens: int = 1024):
        import anthropic  # lazy: only needed on the real path
        self._client = anthropic.Anthropic()
        # Override with any current model via ANTHROPIC_MODEL.
        self._model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
        self._max_tokens = max_tokens

    def create(self, system: str, messages: list[dict], tools: list[dict]):
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=messages,
            tools=tools,
        )
        content = []
        for block in resp.content:
            if block.type == "text":
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content.append({"type": "tool_use", "id": block.id,
                                "name": block.name, "input": block.input})
        return LLMResponse(stop_reason=resp.stop_reason, content=content)
