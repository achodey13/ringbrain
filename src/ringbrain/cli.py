"""Simulate a phone call with RingBrain over the terminal — no Twilio, no
Postgres, no Google credentials needed. Only requires ANTHROPIC_API_KEY.

    python -m ringbrain.cli
"""

import logging
import sys

from ringbrain.agents.graph import build_call_graph
from ringbrain.agents.llm import AnthropicClient
from ringbrain.config import settings
from ringbrain.integrations.calendar import InMemoryCalendarClient
from ringbrain.logging_config import configure_logging
from ringbrain.memory.embedder import SentenceTransformerEmbedder
from ringbrain.memory.inmemory import InMemoryMemoryStore
from ringbrain.nlp.intent import ZeroShotIntentClassifier

configure_logging(level=logging.WARNING)  # keep the REPL clean; warnings (e.g. a caught hallucination) still show


def main() -> None:
    if not settings.anthropic_api_key:
        print("Set ANTHROPIC_API_KEY (in .env or the environment) before running the simulator.")
        sys.exit(1)

    phone = input("Caller phone number (used as customer identity, reused across runs to test memory recall): ").strip()
    phone = phone or "+15555550100"

    print("\nLoading models (first run downloads ~500MB of HF weights)...")
    llm = AnthropicClient(api_key=settings.anthropic_api_key, model=settings.claude_model)
    intent_classifier = ZeroShotIntentClassifier()
    embedder = SentenceTransformerEmbedder()
    memory_store = InMemoryMemoryStore(embedder)
    calendar = InMemoryCalendarClient()

    customer = memory_store.get_or_create_customer(phone)
    graph = build_call_graph(llm, intent_classifier, memory_store, calendar)

    history: list[dict] = []
    print(f"\nRingBrain: Thanks for calling {settings.business_name}. How can I help you today?")
    print("(type 'quit' to hang up)\n")

    while True:
        utterance = input("You: ").strip()
        if utterance.lower() in {"quit", "exit", "hang up"}:
            break

        history.append({"role": "customer", "content": utterance})
        result = graph.invoke(
            {
                "customer_id": customer.id,
                "customer_name": customer.name,
                "business_name": settings.business_name,
                "history": history[:-1],
                "utterance": utterance,
            }
        )
        print(f"RingBrain [{result['intent']}, conf={result['intent_confidence']:.2f}]: {result['reply']}")
        history.append({"role": "agent", "content": result["reply"]})

        if result.get("escalate"):
            print("[call would be transferred to a human here]")
            break
        if result.get("booked_event_id"):
            print(f"[booked calendar event: {result['booked_event_id']} at {result['chosen_slot']}]")

    if history:
        transcript = "\n".join(f"{t['role']}: {t['content']}" for t in history)
        memory_store.remember(customer.id, transcript[:500])
        print("\nCall ended — summary written to this session's in-memory brain.")


if __name__ == "__main__":
    main()
