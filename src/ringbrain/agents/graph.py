from langgraph.graph import END, StateGraph

from ringbrain.agents.llm import LLMClient
from ringbrain.agents.nodes import (
    CalendarClient,
    MemoryReader,
    escalate_node,
    make_booking_node,
    make_check_availability_node,
    make_classify_intent_node,
    make_generate_reply_node,
    make_recall_memory_node,
    route_after_intent,
    route_after_recall,
    route_after_reply,
)
from ringbrain.agents.state import CallState
from ringbrain.nlp.intent import IntentClassifier


def build_call_graph(
    llm: LLMClient,
    intent_classifier: IntentClassifier,
    memory_store: MemoryReader,
    calendar: CalendarClient,
):
    """Wires the RingBrain agent pipeline:

    classify_intent -> [escalate | recall_memory -> (check_availability?) -> generate_reply -> (booking?)]

    Each node is a thin closure over an injected dependency (LLM, calendar,
    memory store, intent classifier), so the whole graph can be built with
    fakes in tests/eval and with real integrations in production.
    """
    graph = StateGraph(CallState)

    graph.add_node("classify_intent", make_classify_intent_node(intent_classifier))
    graph.add_node("recall_memory", make_recall_memory_node(memory_store))
    graph.add_node("check_availability", make_check_availability_node(calendar))
    graph.add_node("generate_reply", make_generate_reply_node(llm))
    graph.add_node("booking", make_booking_node(calendar))
    graph.add_node("escalate", escalate_node)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {"escalate": "escalate", "continue": "recall_memory"},
    )
    graph.add_conditional_edges(
        "recall_memory",
        route_after_recall,
        {"check_availability": "check_availability", "generate_reply": "generate_reply"},
    )
    graph.add_edge("check_availability", "generate_reply")
    graph.add_conditional_edges(
        "generate_reply", route_after_reply, {"booking": "booking", "end": END}
    )
    graph.add_edge("booking", END)
    graph.add_edge("escalate", END)

    return graph.compile()
