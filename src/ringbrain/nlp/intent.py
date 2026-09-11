from dataclasses import dataclass
from typing import Protocol

INTENT_LABELS = [
    "book_appointment",
    "reschedule_appointment",
    "pricing_question",
    "complaint",
    "general_question",
]


@dataclass
class IntentResult:
    intent: str
    confidence: float


class IntentClassifier(Protocol):
    def classify(self, utterance: str) -> IntentResult: ...


class ZeroShotIntentClassifier:
    """Cheap first-pass router: a small zero-shot model tags the caller's
    intent before the LLM agent is invoked, so the expensive model is only
    used for actual reasoning/generation, not for a task a $0 local model
    already handles well.
    """

    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        from transformers import pipeline

        self._pipe = pipeline("zero-shot-classification", model=model_name)

    def classify(self, utterance: str) -> IntentResult:
        result = self._pipe(utterance, candidate_labels=INTENT_LABELS)
        return IntentResult(intent=result["labels"][0], confidence=result["scores"][0])
