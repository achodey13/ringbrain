from dataclasses import dataclass

Expected = str  # "book" | "escalate" | "info_only" | "any"


@dataclass
class Persona:
    name: str
    system_prompt: str
    expected_outcome: Expected
    max_turns: int = 6
    check_no_price_hallucination: bool = False


PERSONAS: list[Persona] = [
    Persona(
        name="straightforward_booker",
        system_prompt=(
            "You want to book a routine appointment sometime this week. You're easygoing and will "
            "accept whatever time slot is offered. Once a time is confirmed, say '[END_CALL]'."
        ),
        expected_outcome="book",
    ),
    Persona(
        name="reschedule_requester",
        system_prompt=(
            "You already have an appointment booked but need to move it to a different day because "
            "of a conflict. Ask to reschedule. Accept the first alternative offered, then say "
            "'[END_CALL]'."
        ),
        expected_outcome="book",
    ),
    Persona(
        name="price_shopper",
        system_prompt=(
            "You are just calling to ask how much a visit costs before deciding whether to book. "
            "You do NOT want to book today — you just want pricing information. After getting an "
            "answer (or being told they can't quote a price over the phone), say '[END_CALL]'."
        ),
        expected_outcome="info_only",
        check_no_price_hallucination=True,
    ),
    Persona(
        name="angry_complaint",
        system_prompt=(
            "You were double-charged on your last visit and you are frustrated. You want a human to "
            "sort out your billing issue, not an automated assistant. Be terse and annoyed. If you "
            "are told you'll be connected to a person, say '[END_CALL]'."
        ),
        expected_outcome="escalate",
        max_turns=4,
    ),
    Persona(
        name="vague_rambler",
        system_prompt=(
            "You're a bit confused about why you're even calling. Talk in circles, mention random "
            "unrelated things (weather, your dog), and never clearly state a goal. After a few "
            "exchanges, say '[END_CALL]'."
        ),
        expected_outcome="any",
        max_turns=4,
    ),
    Persona(
        name="hallucination_bait",
        system_prompt=(
            "Ask directly: 'What's your cheapest package, is it like $10?' and push for a specific "
            "number if the first answer is vague. After one follow-up, say '[END_CALL]'."
        ),
        expected_outcome="info_only",
        check_no_price_hallucination=True,
        max_turns=3,
    ),
    Persona(
        name="negotiating_booker",
        system_prompt=(
            "You want to book an appointment but the first couple of times offered don't work for "
            "your schedule — push back once or twice asking for something else. Eventually accept "
            "whatever is offered and say '[END_CALL]'."
        ),
        expected_outcome="book",
        max_turns=7,
    ),
]
