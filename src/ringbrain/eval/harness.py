"""Runs every persona in eval/personas.py through the real agent graph using an
LLM-simulated caller, then scores task completion, hallucination, and
conversational quality (LLM-as-judge). No Twilio/Postgres/Google needed —
only ANTHROPIC_API_KEY.

    python -m ringbrain.eval.harness
"""

import datetime
import json
import os
import sys

from ringbrain.agents.graph import build_call_graph
from ringbrain.agents.llm import AnthropicClient
from ringbrain.config import settings
from ringbrain.eval.judge import judge_transcript
from ringbrain.eval.metrics import EvalReport, PersonaResult, detect_price_hallucination
from ringbrain.eval.personas import PERSONAS, Persona
from ringbrain.eval.simulate_caller import END_TOKEN, simulate_customer_turn
from ringbrain.integrations.calendar import InMemoryCalendarClient
from ringbrain.memory.embedder import SentenceTransformerEmbedder
from ringbrain.memory.inmemory import InMemoryMemoryStore
from ringbrain.nlp.intent import ZeroShotIntentClassifier

GREETING = "Thanks for calling {business}. How can I help you today?"


def run_persona(llm, intent_classifier, embedder, persona: Persona, business_name: str) -> PersonaResult:
    memory_store = InMemoryMemoryStore(embedder)
    calendar = InMemoryCalendarClient()
    customer = memory_store.get_or_create_customer(f"+1555-eval-{persona.name}")
    graph = build_call_graph(llm, intent_classifier, memory_store, calendar)

    transcript = [{"role": "agent", "content": GREETING.format(business=business_name)}]
    final_outcome = "incomplete"
    agent_replies: list[str] = []

    for _ in range(persona.max_turns):
        customer_line = simulate_customer_turn(llm, persona, transcript)
        if END_TOKEN in customer_line:
            break
        transcript.append({"role": "customer", "content": customer_line})

        result = graph.invoke(
            {
                "customer_id": customer.id,
                "customer_name": customer.name,
                "business_name": business_name,
                "history": transcript[:-1],
                "utterance": customer_line,
            }
        )
        transcript.append({"role": "agent", "content": result["reply"]})
        agent_replies.append(result["reply"])

        if result.get("escalate"):
            final_outcome = "escalate"
            break
        if result.get("booked_event_id"):
            final_outcome = "book"
            break

    if final_outcome == "incomplete" and agent_replies:
        final_outcome = "info_only"

    hallucination = (
        detect_price_hallucination(agent_replies) if persona.check_no_price_hallucination else False
    )
    judge_scores = judge_transcript(llm, transcript)

    return PersonaResult(
        persona_name=persona.name,
        expected_outcome=persona.expected_outcome,
        final_outcome=final_outcome,
        turns_used=len(agent_replies),
        transcript=transcript,
        hallucination_detected=hallucination,
        judge_scores=judge_scores,
    )


def main() -> None:
    if not settings.anthropic_api_key:
        print("Set ANTHROPIC_API_KEY before running the eval harness.")
        sys.exit(1)

    llm = AnthropicClient(api_key=settings.anthropic_api_key, model=settings.claude_model)
    intent_classifier = ZeroShotIntentClassifier()
    embedder = SentenceTransformerEmbedder()

    report = EvalReport()
    for persona in PERSONAS:
        print(f"Running persona: {persona.name}...")
        result = run_persona(llm, intent_classifier, embedder, persona, settings.business_name)
        report.results.append(result)
        status = "PASS" if result.matches_expected else "FAIL"
        print(f"  -> {status} | outcome={result.final_outcome} | turns={result.turns_used}")

    print("\n=== RingBrain Eval Report ===")
    print(f"Task completion rate : {report.task_completion_rate:.0%}")
    print(f"Hallucination rate   : {report.hallucination_rate:.0%}")
    print(f"Average turns/call   : {report.average_turns:.1f}")

    os.makedirs("eval/results", exist_ok=True)
    out_path = f"eval/results/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"\nFull report with transcripts written to {out_path}")


if __name__ == "__main__":
    main()
