from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-5"

    database_url: str = "postgresql+psycopg2://ringbrain:ringbrain@localhost:5432/ringbrain"

    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""

    google_calendar_credentials_path: str = "./credentials/google_calendar_client_secret.json"
    google_calendar_token_path: str = "./credentials/google_calendar_token.json"
    google_calendar_id: str = "primary"

    business_name: str = "Sunrise Dental"
    business_timezone: str = "America/New_York"


settings = Settings()
