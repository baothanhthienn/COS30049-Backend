import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/prompt_injection_db"
    cors_origin: str = "http://localhost:5173"
    model_version: str = "v0.1.0"
    model_dir: str = "models/"
    api_key: str = ""
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    rate_limit_per_minute: int = 60
    inference_concurrency: int = 10
    max_body_bytes: int = 1_048_576
    normalise: bool = True

    @property
    def is_hosted(self) -> bool:
        return any(
            os.getenv(v)
            for v in ("RAILWAY_ENVIRONMENT_NAME", "RAILWAY_ENVIRONMENT", "RAILWAY_PUBLIC_DOMAIN", "RAILWAY_SERVICE_NAME")
        )

    @property
    def database_ssl(self) -> bool:
        return self.is_hosted


settings = Settings()
