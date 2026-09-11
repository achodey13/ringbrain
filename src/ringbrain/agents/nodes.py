import json
from typing import Protocol

from ringbrain.agents.llm import LLMClient
from ringbrain.agents.state import CallState
from ringbrain.memory.store import MemoryStore
from ringbrain.nlp.intent import IntentClassifier

BOOKING_INTENTS = {"book_appointment", "reschedule_appointment"}
ESCALATION_INTENTS = {"complaint"}
CONFIDENCE_THRESHOLD = 0.55


class CalendarClient(Protocol):
    def available_slots(self) -> list[str]: ...
    def book(self, start_time: str, customer_name: str) -> str: ...


def make_classify_intent_node(classifier: IntentClassifier):
    def classify_intent(state: CallState) -> dict:
        result = classifier.classify(state["utterance"])
        return {"intent": result.intent, "intent_confidence": result.confidence}

    return classify_intent


def make_recall_memory_node(memory_store: MemoryStore):
    def recall_memory(state: CallState) -> dict:
        recalled = memory_store.recall(state["customer_id"], state["utterance"], top_k=3)
        return {"recalled_memory": [m.content for m in recalled]}

    return recall_memory


def make_check_availability_node(calendar: CalendarClient):
    def check_availability(state: CallState) -> dict:
        return {"available_slots": calendar.available_slots()}

    return check_availability


def make_generate_reply_node(llm: LLMClient):
    def generate_reply(state: CallState) -> dict:
        system = _build_system_prompt(state)
        transcript = _format_history(state)
        user_message = (
            f"{transcript}\nCustomer: {state['utterance']}\n\n"
            "Respond with ONLY a JSON object, no prose, no markdown fences: "
            '{"reply": "<what you say back to the customer>", '
            '"action": "none" | "book", '
            '"chosen_slot": "<ISO datetime copied exactly from the available slots list, or null>"}'
        )
        raw = llm.complete(system, user_message)
        parsed = _parse_json_response(raw)
        return {
            "reply": parsed["reply"],
            "action": parsed.get("action", "none"),
            "chosen_slot": parsed.get("chosen_slot"),
        }

    return generate_reply


def make_booking_node(calendar: CalendarClient):
    def booking(state: CallState) -> dict:
        if not state.get("chosen_slot"):
            return {"booked_event_id": None}
        event_id = calendar.book(state["chosen_slot"], state.get("customer_name") or "Customer")
        return {"booked_event_id": event_id}

    return booking


def escalate_node(state: CallState) -> dict:
    return {
        "reply": (
            f"I want to make sure this gets handled right — let me connect you with "
            f"someone from {state['business_name']} directly."
        ),
        "escalate": True,
    }


def route_after_intent(state: CallState) -> str:
    if state["intent"] in ESCALATION_INTENTS or state["intent_confidence"] < CONFIDENCE_THRESHOLD:
        return "escalate"
    return "continue"


def route_after_recall(state: CallState) -> str:
    return "check_availability" if state["intent"] in BOOKING_INTENTS else "generate_reply"


def route_after_reply(state: CallState) -> str:
    return "booking" if state.get("action") == "book" and state.get("chosen_slot") else "end"


def _build_system_prompt(state: CallState) -> str:
    memory_block = "\n".join(f"- {m}" for m in state.get("recalled_memory", [])) or "No prior history."
    slots_block = "\n".join(f"- {s}" for s in state.get("available_slots", [])) or "N/A"
    return (
        f"You are the phone/text receptionist for {state['business_name']}. "
        "Be warm, concise, and never invent information that isn't given to you below "
        "(pricing, availability, or facts about the customer)."
        f"\n\nWhat you remember about this customer:\n{memory_block}"
        f"\n\nCurrently open appointment slots (only offer these, verbatim):\n{slots_block}"
    )


def _format_history(state: CallState) -> str:
    lines = []
    for turn in state.get("history", []):
        speaker = "Customer" if turn["role"] == "customer" else "Agent"
        lines.append(f"{speaker}: {turn['content']}")
    return "\n".join(lines)


def _parse_json_response(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)
