import logging
from typing import Protocol

import anthropic
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger("ringbrain.llm")

RETRYABLE_ERRORS = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)


class LLMClient(Protocol):
    def complete(self, system: str, user_message: str) -> str:
        """Return the raw text completion for a single-turn prompt."""
        ...


class AnthropicClient:
    """Claude is on the critical path of every call — a transient network
    blip or rate limit shouldn't drop a live customer. Retries with backoff
    on connection/rate-limit/server errors only; a bad-request error (4xx
    other than 429) fails fast instead of retrying something that will never
    succeed.
    """

    def __init__(self, api_key: str, model: str):
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    @retry(
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def complete(self, system: str, user_message: str) -> str:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=512,
                system=system,
                messages=[{"role": "user", "content": user_message}],
            )
        except RETRYABLE_ERRORS as exc:
            logger.warning("Claude call failed, will retry: %s", exc)
            raise

        block = response.content[0]
        if not isinstance(block, anthropic.types.TextBlock):
            raise ValueError(f"Expected a text response block, got {type(block).__name__}")
        return block.text
