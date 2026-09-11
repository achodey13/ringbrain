from fastapi import FastAPI, Form, Response
from twilio.twiml.messaging_response import MessagingResponse

from ringbrain.agents.graph import build_call_graph
from ringbrain.api.runtime import get_runtime
from ringbrain.config import settings
from ringbrain.db.models import Channel, Conversation, Message
from ringbrain.db.session import get_session
from ringbrain.integrations.twilio_voice import greeting_twiml, reply_twiml
from ringbrain.memory.store import MemoryStore

app = FastAPI(title="RingBrain")


@app.get("/health")
def health():
    return {"status": "ok"}


def _run_turn(customer_phone: str, utterance: str, history: list[dict]) -> dict:
    """Runs one turn of the agent graph against a fresh DB-backed memory store."""
    runtime = get_runtime()
    with get_session() as db:
        memory_store = MemoryStore(db, runtime.embedder)
        customer = memory_store.get_or_create_customer(customer_phone)
        graph = build_call_graph(runtime.llm, runtime.intent_classifier, memory_store, runtime.calendar)
        result = graph.invoke(
            {
                "customer_id": customer.id,
                "customer_name": customer.name,
                "business_name": settings.business_name,
                "history": history,
                "utterance": utterance,
            }
        )
        result["customer_id"] = customer.id
        return result


def _persist_conversation(customer_id: str, channel: Channel, history: list[dict]) -> None:
    runtime = get_runtime()
    with get_session() as db:
        memory_store = MemoryStore(db, runtime.embedder)
        transcript = "\n".join(
            f"{'Customer' if t['role'] == 'customer' else 'Agent'}: {t['content']}" for t in history
        )
        summary = runtime.summarizer.summarize(transcript) if len(history) >= 2 else transcript

        conversation = Conversation(customer_id=customer_id, channel=channel, summary=summary)
        db.add(conversation)
        db.flush()
        for turn in history:
            db.add(Message(conversation_id=conversation.id, role=turn["role"], content=turn["content"]))

        memory_store.remember(customer_id, summary, conversation_id=conversation.id)


@app.post("/voice/incoming")
def voice_incoming(CallSid: str = Form(...), From: str = Form(...)):
    runtime = get_runtime()
    with get_session() as db:
        memory_store = MemoryStore(db, runtime.embedder)
        customer = memory_store.get_or_create_customer(From)
        runtime.sessions.get_or_create(CallSid, customer.id)

    gather_url = "/voice/gather"
    twiml = greeting_twiml(settings.business_name, gather_url)
    return Response(content=twiml, media_type="application/xml")


@app.post("/voice/gather")
def voice_gather(CallSid: str = Form(...), From: str = Form(...), SpeechResult: str = Form("")):
    runtime = get_runtime()
    session = runtime.sessions.get_or_create(CallSid, customer_id="")

    if not SpeechResult:
        twiml = reply_twiml("Sorry, I didn't catch that — could you say that again?", "/voice/gather", hang_up=False)
        return Response(content=twiml, media_type="application/xml")

    session.history.append({"role": "customer", "content": SpeechResult})
    result = _run_turn(From, SpeechResult, session.history[:-1])
    session.history.append({"role": "agent", "content": result["reply"]})

    call_over = bool(result.get("escalate")) or bool(result.get("booked_event_id"))
    if call_over:
        _persist_conversation(result["customer_id"], Channel.voice, session.history)
        runtime.sessions.pop(CallSid)

    twiml = reply_twiml(result["reply"], "/voice/gather", hang_up=call_over)
    return Response(content=twiml, media_type="application/xml")


@app.post("/sms/incoming")
def sms_incoming(From: str = Form(...), Body: str = Form(...)):
    result = _run_turn(From, Body, history=[])
    _persist_conversation(
        result["customer_id"],
        Channel.sms,
        [{"role": "customer", "content": Body}, {"role": "agent", "content": result["reply"]}],
    )

    response = MessagingResponse()
    response.message(result["reply"])
    return Response(content=str(response), media_type="application/xml")
