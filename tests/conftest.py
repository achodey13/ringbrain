import pytest

from ringbrain.integrations.calendar import InMemoryCalendarClient
from ringbrain.nlp.intent import IntentResult


class FakeEmbedder:
    """Deterministic, dependency-free stand-in for the real sentence-transformers
    embedder so memory tests don't need to download a model or hit the network.
    """

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * 16
        for ch in text.lower():
            vec[ord(ch) % 16] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]


class ScriptedIntentClassifier:
    def __init__(self, intent: str, confidence: float = 0.9):
        self.intent = intent
        self.confidence = confidence

    def classify(self, utterance: str) -> IntentResult:
        return IntentResult(intent=self.intent, confidence=self.confidence)


class ScriptedLLM:
    """Stands in for AnthropicClient: returns queued responses instead of
    calling a real model, so agent-graph tests are fast, free, and offline.
    """

    def __init__(self, responses: list[str]):
        self._responses = list(responses)

    def complete(self, system: str, user_message: str) -> str:
        if len(self._responses) == 1:
            return self._responses[0]
        return self._responses.pop(0)


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def fake_calendar():
    return InMemoryCalendarClient(slots=["2026-01-01T10:00:00+00:00", "2026-01-01T14:00:00+00:00"])
