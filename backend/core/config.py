from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    collection_name: str = "rag_collection"
    embedding_model: str = "jina-embeddings-v4"
    jina_api_key: SecretStr | None = None
    phoenix_url: str = "http://localhost:6006/v1/traces"
    openrouter_api_key: SecretStr | None = None
    openrouter_model: str = "openrouter/free"
    ragas_eval_model: str = "openrouter/free"
    database_url: str | None = None
    cors_origins: list[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_openrouter_api_key(self):
        if self.openrouter_api_key is None:
            raise ValueError("OPENROUTER_API_KEY is required")

        if self.openrouter_api_key.get_secret_value().strip() == "":
            raise ValueError("OPENROUTER_API_KEY cannot be empty")

        return self
