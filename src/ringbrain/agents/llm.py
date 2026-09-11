from typing import Protocol


class LLMClient(Protocol):
    def complete(self, system: str, user_message: str) -> str:
        """Return the raw text completion for a single-turn prompt."""
        ...


class AnthropicClient:
    def __init__(self, api_key: str, model: str):
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key)
        self._model = model

    def complete(self, system: str, user_message: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text
