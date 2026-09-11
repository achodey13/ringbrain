from dataclasses import dataclass, field

from ringbrain.agents.llm import AnthropicClient, LLMClient
from ringbrain.agents.state import Turn
from ringbrain.config import settings
from ringbrain.integrations.calendar import GoogleCalendarClient
from ringbrain.integrations.sms import SMSClient, TwilioSMSClient
from ringbrain.memory.embedder import Embedder, SentenceTransformerEmbedder
from ringbrain.nlp.intent import IntentClassifier, ZeroShotIntentClassifier
from ringbrain.nlp.summarizer import DialogueSummarizer, Summarizer


@dataclass
class CallSession:
    customer_id: str
    history: list[Turn] = field(default_factory=list)


class CallSessionStore:
    """Keeps in-flight call state (history) keyed by Twilio CallSid, between
    the stateless HTTP webhook requests Twilio makes for each conversation
    turn. A single-process dict is fine for a portfolio deployment; swap for
    Redis to run more than one API worker.
    """

    def __init__(self):
        self._sessions: dict[str, CallSession] = {}

    def get_or_create(self, call_sid: str, customer_id: str) -> CallSession:
        if call_sid not in self._sessions:
            self._sessions[call_sid] = CallSession(customer_id=customer_id)
        return self._sessions[call_sid]

    def pop(self, call_sid: str) -> CallSession | None:
        return self._sessions.pop(call_sid, None)


@dataclass
class Runtime:
    llm: LLMClient
    intent_classifier: IntentClassifier
    embedder: Embedder
    summarizer: Summarizer
    calendar: "object"
    sms: SMSClient
    sessions: CallSessionStore


_runtime: Runtime | None = None


def get_runtime() -> Runtime:
    global _runtime
    if _runtime is None:
        _runtime = Runtime(
            llm=AnthropicClient(api_key=settings.anthropic_api_key, model=settings.claude_model),
            intent_classifier=ZeroShotIntentClassifier(),
            embedder=SentenceTransformerEmbedder(),
            summarizer=DialogueSummarizer(),
            calendar=GoogleCalendarClient(),
            sms=TwilioSMSClient(
                settings.twilio_account_sid, settings.twilio_auth_token, settings.twilio_from_number
            ),
            sessions=CallSessionStore(),
        )
    return _runtime
