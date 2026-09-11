import json

from ringbrain.agents.graph import build_call_graph
from ringbrain.memory.inmemory import InMemoryMemoryStore
from tests.conftest import ScriptedIntentClassifier, ScriptedLLM


def _base_state(customer, utterance, business_name="Test Biz"):
    return {
        "customer_id": customer.id,
        "customer_name": customer.name,
        "business_name": business_name,
        "history": [],
        "utterance": utterance,
    }


def test_booking_flow(fake_embedder, fake_calendar):
    memory_store = InMemoryMemoryStore(fake_embedder)
    customer = memory_store.get_or_create_customer("+15551234567")

    llm = ScriptedLLM(
        [
            json.dumps(
                {
                    "reply": "You're booked for 10am on Jan 1st!",
                    "action": "book",
                    "chosen_slot": fake_calendar.available_slots()[0],
                }
            )
        ]
    )
    intent_classifier = ScriptedIntentClassifier("book_appointment", confidence=0.95)

    graph = build_call_graph(llm, intent_classifier, memory_store, fake_calendar)
    result = graph.invoke(_base_state(customer, "I'd like to book an appointment"))

    assert result["action"] == "book"
    assert result["booked_event_id"] is not None
    assert len(fake_calendar.booked) == 1


def test_escalation_flow_on_complaint_intent(fake_embedder, fake_calendar):
    memory_store = InMemoryMemoryStore(fake_embedder)
    customer = memory_store.get_or_create_customer("+15559876543")

    llm = ScriptedLLM(["should not be called"])
    intent_classifier = ScriptedIntentClassifier("complaint", confidence=0.9)

    graph = build_call_graph(llm, intent_classifier, memory_store, fake_calendar)
    result = graph.invoke(_base_state(customer, "I was overcharged and I'm furious"))

    assert result["escalate"] is True
    assert len(fake_calendar.booked) == 0


def test_escalation_flow_on_low_confidence(fake_embedder, fake_calendar):
    memory_store = InMemoryMemoryStore(fake_embedder)
    customer = memory_store.get_or_create_customer("+15550001111")

    llm = ScriptedLLM(["should not be called"])
    intent_classifier = ScriptedIntentClassifier("general_question", confidence=0.2)

    graph = build_call_graph(llm, intent_classifier, memory_store, fake_calendar)
    result = graph.invoke(_base_state(customer, "uh, I don't know, something about my dog"))

    assert result["escalate"] is True


def test_info_only_flow_skips_availability_check(fake_embedder, fake_calendar):
    memory_store = InMemoryMemoryStore(fake_embedder)
    customer = memory_store.get_or_create_customer("+15552223333")

    llm = ScriptedLLM(
        [json.dumps({"reply": "We're open 9-5 Monday through Friday.", "action": "none", "chosen_slot": None})]
    )
    intent_classifier = ScriptedIntentClassifier("general_question", confidence=0.8)

    graph = build_call_graph(llm, intent_classifier, memory_store, fake_calendar)
    result = graph.invoke(_base_state(customer, "What are your hours?"))

    assert result["action"] == "none"
    assert not result.get("available_slots")  # availability-check node was skipped entirely
    assert result.get("booked_event_id") is None


def test_malformed_llm_response_escalates_instead_of_crashing(fake_embedder, fake_calendar):
    memory_store = InMemoryMemoryStore(fake_embedder)
    customer = memory_store.get_or_create_customer("+15556667777")

    llm = ScriptedLLM(["not valid json at all"])
    intent_classifier = ScriptedIntentClassifier("general_question", confidence=0.8)

    graph = build_call_graph(llm, intent_classifier, memory_store, fake_calendar)
    result = graph.invoke(_base_state(customer, "What are your hours?"))

    assert result["escalate"] is True
    assert result["action"] == "none"


def test_hallucinated_slot_is_downgraded_not_booked(fake_embedder, fake_calendar):
    memory_store = InMemoryMemoryStore(fake_embedder)
    customer = memory_store.get_or_create_customer("+15558889999")

    llm = ScriptedLLM(
        [
            json.dumps(
                {
                    "reply": "You're all set for 3pm tomorrow!",
                    "action": "book",
                    "chosen_slot": "2099-01-01T15:00:00+00:00",  # not a real available slot
                }
            )
        ]
    )
    intent_classifier = ScriptedIntentClassifier("book_appointment", confidence=0.95)

    graph = build_call_graph(llm, intent_classifier, memory_store, fake_calendar)
    result = graph.invoke(_base_state(customer, "Book me for tomorrow"))

    assert result["action"] == "none"
    assert result["chosen_slot"] is None
    assert result.get("booked_event_id") is None
    assert len(fake_calendar.booked) == 0


def test_memory_recall_finds_relevant_past_context(fake_embedder):
    memory_store = InMemoryMemoryStore(fake_embedder)
    customer = memory_store.get_or_create_customer("+15554445555")
    memory_store.remember(customer.id, "Customer previously asked about teeth whitening pricing.")
    memory_store.remember(customer.id, "Customer's dog is named Max.")

    recalled = memory_store.recall(customer.id, "Do you have any info on whitening?", top_k=1)

    assert recalled
    assert "whitening" in recalled[0].content
