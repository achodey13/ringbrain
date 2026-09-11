import json

from ringbrain.agents.llm import LLMClient

JUDGE_SYSTEM_PROMPT = """You are grading a transcript between an AI phone receptionist ("Agent") \
and a customer. Score the Agent's performance on three axes, each 1-5 (5 is best):

- naturalness: does the Agent sound like a helpful human receptionist, not a robotic script?
- helpfulness: did the Agent make real progress on what the customer needed?
- faithfulness: did the Agent avoid inventing facts (prices, availability, policies) it wasn't \
given in the system context?

Respond with ONLY a JSON object: \
{"naturalness": <int>, "helpfulness": <int>, "faithfulness": <int>, "rationale": "<one sentence>"}"""


def judge_transcript(llm: LLMClient, transcript: list[dict]) -> dict:
    formatted = "\n".join(
        f"{'Customer' if t['role'] == 'customer' else 'Agent'}: {t['content']}" for t in transcript
    )
    raw = llm.complete(JUDGE_SYSTEM_PROMPT, formatted).strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)
