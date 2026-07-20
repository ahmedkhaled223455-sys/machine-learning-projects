"""LLM client abstraction (same pattern as the llm-agent project).

`AnthropicLLM` calls Claude; `MockLLM` returns scripted text so the
grounded-generation step is testable with no API key. The RAG pipeline
only needs plain text completion here (the retrieval does the heavy
lifting), so the interface is a single `complete(system, user)` call.
"""

from __future__ import annotations

import os


class MockLLM:
    """Returns scripted completions in order -- for tests/offline demos."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        if not self._responses:
            raise AssertionError("MockLLM ran out of scripted responses")
        return self._responses.pop(0)


class AnthropicLLM:
    """Real Claude client. Requires `anthropic` and ANTHROPIC_API_KEY."""

    def __init__(self, model: str | None = None, max_tokens: int = 512):
        import anthropic  # lazy import: only needed on the real path
        self._client = anthropic.Anthropic()
        self._model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
        self._max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")
