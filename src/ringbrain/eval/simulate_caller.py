from ringbrain.agents.llm import LLMClient
from ringbrain.eval.personas import Persona

END_TOKEN = "[END_CALL]"


def simulate_customer_turn(llm: LLMClient, persona: Persona, transcript: list[dict]) -> str:
    """Uses an LLM to role-play the persona and produce the next thing the
    caller says, given the conversation so far. This is what lets the eval
    harness run realistic multi-turn conversations without a human tester.
    """
    system = (
        "You are role-playing as a customer calling a small local business. "
        f"Your persona and goal: {persona.system_prompt}\n"
        "Reply with ONLY the next line you say out loud — one or two sentences of natural "
        f"spoken language. No stage directions, no quotation marks. When you're done with the "
        f"call, output exactly {END_TOKEN} and nothing else."
    )
    if not transcript:
        user_message = "The receptionist just answered and greeted you. What do you say?"
    else:
        convo = "\n".join(
            f"{'You' if t['role'] == 'customer' else 'Receptionist'}: {t['content']}" for t in transcript
        )
        user_message = f"{convo}\n\nWhat do you say next?"

    return llm.complete(system, user_message).strip()
