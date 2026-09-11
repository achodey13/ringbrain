# RingBrain

A voice/SMS AI receptionist that answers calls, qualifies leads, books real appointments, and — unlike a plain chatbot wrapper — **remembers every customer** across future contacts via a persistent semantic memory layer.

Built as a portfolio project demonstrating production AI-engineering patterns: multi-agent orchestration, tool-calling into real external systems, retrieval-based memory, and eval-driven development — not just a prompt pasted into an API call.

## What it does

A caller phones (or texts) a small business. RingBrain:
1. Classifies intent with a lightweight local model before ever invoking the LLM (cheap router, expensive reasoning only when needed)
2. Recalls relevant history about that customer from a vector-backed memory store
3. Routes low-confidence or complaint-flagged calls straight to human escalation
4. Checks real calendar availability and books the appointment via the Google Calendar API
5. Writes a summary of the call back into the customer's permanent memory

```
                        ┌─────────────────┐
   utterance ─────────► │ classify_intent  │  (HF zero-shot, local, free)
                        └────────┬─────────┘
                                 │
              low confidence /   │  confident
              complaint          ▼
                 │        ┌─────────────┐
                 │        │recall_memory│  (pgvector semantic search)
                 ▼        └──────┬──────┘
            ┌──────────┐         │
            │ escalate │    booking intent?
            └──────────┘         │
                            ┌─────┴─────┐
                            ▼           ▼
                  check_availability  (skip)
                    (Google Calendar)   │
                            └─────┬─────┘
                                  ▼
                          ┌───────────────┐
                          │ generate_reply│  (Claude — reasoning/dialogue)
                          └───────┬───────┘
                                  │ action == book?
                                  ▼
                             ┌─────────┐
                             │ booking │  (writes real calendar event)
                             └─────────┘
```

Graph implemented in [`src/ringbrain/agents/graph.py`](src/ringbrain/agents/graph.py) with [LangGraph](https://github.com/langchain-ai/langgraph).

## Why this design

- **Small model as router, big model as reasoner.** Intent classification uses a local zero-shot HF model (`facebook/bart-large-mnli`), not Claude — cheaper, faster, and it demonstrates knowing when *not* to reach for an LLM.
- **Memory is retrieval, not a growing prompt.** Every past conversation is summarized and embedded (`sentence-transformers/all-MiniLM-L6-v2`) into Postgres/pgvector, then semantically searched per-call — this scales to years of history without blowing the context window.
- **A confidence gate, not blind automation.** Complaints and low-confidence intents never reach the booking/reply logic — they're routed to human handoff. This is the difference between a toy demo and something you'd actually trust with real customers.
- **Every dependency is behind an interface.** `LLMClient`, `Embedder`, `IntentClassifier`, `CalendarClient`, `SMSClient` are all `Protocol`s with a real implementation and a fake. The entire agent graph, eval harness, and CLI simulator run with zero external services — only `ANTHROPIC_API_KEY` is required to actually talk to it.

## Project layout

```
src/ringbrain/
  agents/        LangGraph state machine, nodes, Claude client
  memory/        pgvector-backed semantic memory + in-memory fake
  nlp/           HF zero-shot intent classifier, HF dialogue summarizer
  integrations/  Google Calendar, Twilio voice (TwiML) + SMS
  db/            SQLAlchemy models (Customer, Conversation, Message, Appointment, MemoryEntry)
  api/           FastAPI app — Twilio webhook endpoints
  eval/          persona-based simulated-caller eval harness + LLM-as-judge
  cli.py         talk to RingBrain in your terminal, no infra required
```

## Quickstart (no Twilio/Google/Postgres needed)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # then set ANTHROPIC_API_KEY

python -m ringbrain.cli
```

This runs the full agent graph — intent classification, memory recall, an in-memory fake calendar — entirely in your terminal. Use the same phone number across two runs to see it recall prior context.

## Run the eval suite

```bash
python -m ringbrain.eval.harness
```

Runs 7 synthetic caller personas (straightforward booking, rescheduling, price-shopping, an angry complaint, a rambling/ambiguous caller, a hallucination-bait prompt, and a negotiating booker) end-to-end against the real agent graph, using an LLM to *play* the caller. Scores:

- **Task completion rate** — did the call end in the outcome the persona expected (booked / escalated / info-only)?
- **Hallucination rate** — did the agent ever invent a price it was never given?
- **LLM-as-judge** — naturalness / helpfulness / faithfulness, 1–5, on every transcript

Full transcripts + scores are written to `eval/results/<timestamp>.json`.

## Run the unit tests

```bash
pytest
```

All tests use fakes (`ScriptedLLM`, `FakeEmbedder`, `ScriptedIntentClassifier`, `InMemoryCalendarClient`) — no API key, network access, or database required. Covers the graph's routing logic (booking, escalation on complaint, escalation on low confidence, info-only skipping the calendar check) and memory recall.

## Going live (optional — requires your own accounts)

The webhook layer (`src/ringbrain/api/main.py`) and integrations are real, working code — they just need credentials this repo can't ship with:

1. **Postgres + pgvector**: `docker compose up -d`, then `python -c "from ringbrain.db.session import init_db; init_db()"`
2. **Google Calendar**: create an OAuth client ID in Google Cloud Console, download the JSON, point `GOOGLE_CALENDAR_CREDENTIALS_PATH` at it, run the auth flow once
3. **Twilio**: buy/use a trial number, set `TWILIO_*` env vars, point the number's voice webhook at `POST /voice/incoming` and SMS webhook at `POST /sms/incoming` (via `ngrok` for local testing)
4. `uvicorn ringbrain.api.main:app --reload`

## Known simplifications (and what production would add)

- Call-session state lives in an in-process dict (`CallSessionStore`) — fine for one API worker, would move to Redis for horizontal scaling
- "Call over" is inferred from escalation/booking rather than a full dialogue-state tracker
- Speech-to-text/text-to-speech use Twilio's built-in speech recognition and `<Say>` rather than a streaming Whisper/TTS pipeline — swappable later for lower latency without touching the agent graph
- No Alembic migrations — `init_db()` does a straight `create_all` for a portfolio-scale project

## Resume bullets

- Built a multi-agent voice/SMS AI receptionist (LangGraph + Claude) with tool-calling into live Google Calendar and Twilio APIs, gated by a confidence-based human-escalation path
- Designed a persistent semantic memory layer (pgvector + sentence-transformers) giving the agent cross-conversation recall of customer history
- Built an automated eval harness with LLM-simulated callers across 7 personas, tracking task-completion rate, hallucination rate, and LLM-judged conversational quality — enabling regression detection on prompt/model changes
- Applied a router pattern combining a free local zero-shot classifier for intent detection with an LLM only for generation, cutting unnecessary model calls
