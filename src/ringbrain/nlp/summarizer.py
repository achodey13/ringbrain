from typing import Protocol


class Summarizer(Protocol):
    def summarize(self, transcript: str) -> str: ...


class DialogueSummarizer:
    """Summarizes a call/text transcript into a short note for the customer
    record, using a model fine-tuned specifically for dialogue (not generic
    news summarization) since call transcripts are conversational, not prose.
    """

    def __init__(self, model_name: str = "philschmid/bart-large-cnn-samsum"):
        from transformers import pipeline

        self._pipe = pipeline("summarization", model=model_name)  # type: ignore[call-overload]

    def summarize(self, transcript: str) -> str:
        result = self._pipe(transcript, max_length=80, min_length=10, do_sample=False)
        return result[0]["summary_text"]
