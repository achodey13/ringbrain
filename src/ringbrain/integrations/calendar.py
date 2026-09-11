import datetime
import os

from ringbrain.config import settings

SCOPES = ["https://www.googleapis.com/auth/calendar"]
SLOT_DURATION_MINUTES = 30
BUSINESS_HOURS = (9, 17)  # 9am-5pm
LOOKAHEAD_DAYS = 5


class GoogleCalendarClient:
    """Real Google Calendar integration. Requires a one-time OAuth flow:

    1. Create an OAuth client ID (Desktop app) in Google Cloud Console with
       the Calendar API enabled, download the JSON, save it to
       GOOGLE_CALENDAR_CREDENTIALS_PATH.
    2. Run `python -m ringbrain.cli auth-calendar` once to complete the
       browser consent flow; the resulting token is cached at
       GOOGLE_CALENDAR_TOKEN_PATH and refreshed automatically after that.
    """

    def __init__(self):
        self._service = None  # lazily built — avoids importing google libs / requiring auth in tests

    def _get_service(self):
        if self._service is not None:
            return self._service

        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build

        creds = None
        token_path = settings.google_calendar_token_path
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    settings.google_calendar_credentials_path, SCOPES
                )
                creds = flow.run_local_server(port=0)
            os.makedirs(os.path.dirname(token_path) or ".", exist_ok=True)
            with open(token_path, "w") as f:
                f.write(creds.to_json())

        self._service = build("calendar", "v3", credentials=creds)
        return self._service

    def available_slots(self) -> list[str]:
        service = self._get_service()
        now = datetime.datetime.now(datetime.UTC)
        time_min = now.isoformat()
        time_max = (now + datetime.timedelta(days=LOOKAHEAD_DAYS)).isoformat()

        busy = (
            service.freebusy()
            .query(
                body={
                    "timeMin": time_min,
                    "timeMax": time_max,
                    "items": [{"id": settings.google_calendar_id}],
                }
            )
            .execute()["calendars"][settings.google_calendar_id]["busy"]
        )
        busy_ranges = [
            (datetime.datetime.fromisoformat(b["start"]), datetime.datetime.fromisoformat(b["end"]))
            for b in busy
        ]

        slots = []
        day = now.date()
        for _ in range(LOOKAHEAD_DAYS):
            for hour in range(*BUSINESS_HOURS):
                for minute in (0, 30):
                    start = datetime.datetime.combine(
                        day, datetime.time(hour, minute), tzinfo=datetime.UTC
                    )
                    if start <= now:
                        continue
                    end = start + datetime.timedelta(minutes=SLOT_DURATION_MINUTES)
                    if not any(start < b_end and end > b_start for b_start, b_end in busy_ranges):
                        slots.append(start.isoformat())
            day += datetime.timedelta(days=1)
        return slots[:10]

    def book(self, start_time: str, customer_name: str) -> str:
        service = self._get_service()
        start = datetime.datetime.fromisoformat(start_time)
        end = start + datetime.timedelta(minutes=SLOT_DURATION_MINUTES)
        event = (
            service.events()
            .insert(
                calendarId=settings.google_calendar_id,
                body={
                    "summary": f"Appointment — {customer_name}",
                    "start": {"dateTime": start.isoformat()},
                    "end": {"dateTime": end.isoformat()},
                },
            )
            .execute()
        )
        return event["id"]


class InMemoryCalendarClient:
    """Fake calendar used by the CLI simulator and eval harness so RingBrain
    is fully runnable/testable without Google credentials.
    """

    def __init__(self, slots: list[str] | None = None):
        now = datetime.datetime.now(datetime.UTC).replace(
            minute=0, second=0, microsecond=0
        )
        self._slots = slots or [
            (now + datetime.timedelta(days=d, hours=h)).isoformat()
            for d in (1, 1, 2, 2, 3)
            for h in (10, 14)
        ][:5]
        self.booked: list[tuple[str, str]] = []

    def available_slots(self) -> list[str]:
        return list(self._slots)

    def book(self, start_time: str, customer_name: str) -> str:
        if start_time in self._slots:
            self._slots.remove(start_time)
        self.booked.append((start_time, customer_name))
        return f"fake-event-{len(self.booked)}"
