import json

import pytest

from ringbrain.agents.nodes import FALLBACK_REPLY, make_generate_reply_node, route_after_reply
from ringbrain.agents.state import CallState

AVAILABLE_SLOT = "2026-01-01T10:00:00+00:00"
BOOKING_CONFIRMATION = "You're all set for your appointment!"


class FixedResponseLLM:
    def __init__(self, response: object):
        self.response = response

    def complete(self, system: str, user_message: str) -> str:
        return json.dumps(self.response)


def _state(slots: list[str]) -> CallState:
    return {
        "business_name": "Test Biz",
        "utterance": "Please book an appointment",
        "history": [],
        "available_slots": slots,
    }


@pytest.mark.parametrize("chosen_slot", ["2099-01-01T15:00:00+00:00", None, ""])
def test_unavailable_booking_replaces_confirmation_with_handoff(chosen_slot):
    node = make_generate_reply_node(
        FixedResponseLLM(
            {"reply": BOOKING_CONFIRMATION, "action": "book", "chosen_slot": chosen_slot}
        )
    )

    result = node(_state([AVAILABLE_SLOT]))

    assert result == {
        "reply": FALLBACK_REPLY,
        "action": "none",
        "chosen_slot": None,
        "escalate": True,
    }
    assert route_after_reply(result) == "end"


def test_booking_without_calendar_availability_does_not_confirm():
    node = make_generate_reply_node(
        FixedResponseLLM(
            {"reply": BOOKING_CONFIRMATION, "action": "book", "chosen_slot": AVAILABLE_SLOT}
        )
    )

    result = node(_state([]))

    assert result["reply"] == FALLBACK_REPLY
    assert result["escalate"] is True
    assert route_after_reply(result) == "end"


def test_available_booking_still_routes_to_calendar():
    node = make_generate_reply_node(
        FixedResponseLLM(
            {"reply": BOOKING_CONFIRMATION, "action": "book", "chosen_slot": AVAILABLE_SLOT}
        )
    )

    result = node(_state([AVAILABLE_SLOT]))

    assert result == {
        "reply": BOOKING_CONFIRMATION,
        "action": "book",
        "chosen_slot": AVAILABLE_SLOT,
    }
    assert route_after_reply(result) == "booking"


def test_information_reply_still_ends_without_handoff():
    node = make_generate_reply_node(
        FixedResponseLLM({"reply": "We're open 9-5.", "action": "none", "chosen_slot": None})
    )

    result = node(_state([]))

    assert result == {"reply": "We're open 9-5.", "action": "none", "chosen_slot": None}
    assert route_after_reply(result) == "end"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reply", None),
        ("reply", True),
        ("reply", 12),
        ("reply", ["Booked"]),
        ("reply", {"text": "Booked"}),
        ("reply", ""),
        ("reply", " \n\t"),
        ("action", None),
        ("action", False),
        ("action", 1),
        ("action", []),
        ("action", {}),
        ("action", "cancel"),
        ("action", "BOOK"),
        ("chosen_slot", []),
        ("chosen_slot", {}),
        ("chosen_slot", 123),
        ("chosen_slot", True),
    ],
)
def test_invalid_response_fields_escalate_without_using_model_reply(field, value):
    response = {"reply": BOOKING_CONFIRMATION, "action": "none", "chosen_slot": None}
    response[field] = value
    node = make_generate_reply_node(FixedResponseLLM(response))

    result = node(_state([AVAILABLE_SLOT]))

    assert result == {
        "reply": FALLBACK_REPLY,
        "action": "none",
        "chosen_slot": None,
        "escalate": True,
    }
    assert route_after_reply(result) == "end"


@pytest.mark.parametrize("response", [None, [], "Booked", 12, True])
def test_non_object_json_response_escalates(response):
    node = make_generate_reply_node(FixedResponseLLM(response))

    result = node(_state([AVAILABLE_SLOT]))

    assert result == {
        "reply": FALLBACK_REPLY,
        "action": "none",
        "chosen_slot": None,
        "escalate": True,
    }
    assert route_after_reply(result) == "end"


def test_information_response_keeps_optional_field_defaults_and_reply_text():
    reply = " We're open 9-5. "
    node = make_generate_reply_node(FixedResponseLLM({"reply": reply}))

    result = node(_state([]))

    assert result == {"reply": reply, "action": "none", "chosen_slot": None}
    assert route_after_reply(result) == "end"
