from typing import Protocol


class SMSClient(Protocol):
    def send(self, to: str, body: str) -> str: ...


class TwilioSMSClient:
    def __init__(self, account_sid: str, auth_token: str, from_number: str):
        from twilio.rest import Client

        self._client = Client(account_sid, auth_token)
        self._from_number = from_number

    def send(self, to: str, body: str) -> str:
        message = self._client.messages.create(to=to, from_=self._from_number, body=body)
        return message.sid


class InMemorySMSClient:
    """Fake SMS sender for local/eval use — records what would have been sent."""

    def __init__(self):
        self.sent: list[tuple[str, str]] = []

    def send(self, to: str, body: str) -> str:
        self.sent.append((to, body))
        return f"fake-sms-{len(self.sent)}"
