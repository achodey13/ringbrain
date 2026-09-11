from typing import TypedDict


class Turn(TypedDict):
    role: str  # "customer" | "agent"
    content: str


class CallState(TypedDict, total=False):
    # inputs
    customer_id: str
    customer_name: str | None
    business_name: str
    history: list[Turn]
    utterance: str

    # populated by nodes
    intent: str
    intent_confidence: float
    recalled_memory: list[str]
    available_slots: list[str]
    reply: str
    action: str  # "none" | "book" | "escalate"
    chosen_slot: str | None
    booked_event_id: str | None
    escalate: bool
