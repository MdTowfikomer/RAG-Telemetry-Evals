from typing import Any
from pydantic import SecretStr, model_validator, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    collection_name: str = "rag_collection"
    embedding_model: str = "embed-english-v3.0"
    cohere_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    openrouter_model: str = "openrouter/free"
    ragas_eval_model: str = "openrouter/free"
    database_url: str | None = None
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "https://rag-telemetry-evals.onrender.com",
        "https://ragevals.vercel.app",
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            # Try to parse as JSON list
            try:
                import json
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed]
            except json.JSONDecodeError:
                pass
            # Split comma-separated string
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @model_validator(mode="after")
    def validate_openrouter_api_key(self):
        if self.openrouter_api_key is None:
            raise ValueError("OPENROUTER_API_KEY is required")

        if self.openrouter_api_key.get_secret_value().strip() == "":
            raise ValueError("OPENROUTER_API_KEY cannot be empty")

        return self
