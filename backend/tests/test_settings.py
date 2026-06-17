import os
import unittest
from unittest.mock import patch

from pydantic import SecretStr, ValidationError
from pydantic_settings import SettingsConfigDict

from backend.core import Settings


class NoEnvFileSettings(Settings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")


class TestSettings(unittest.TestCase):
    def test_missing_openrouter_api_key_fails_fast(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValidationError):
                NoEnvFileSettings()

    def test_empty_openrouter_api_key_fails(self):
        with self.assertRaises(ValidationError):
            NoEnvFileSettings(openrouter_api_key=SecretStr(""))

    def test_explicit_openrouter_api_key_is_accepted(self):
        settings = NoEnvFileSettings(
            openrouter_api_key=SecretStr("test-key"),
            qdrant_url="http://localhost:7000",
            collection_name="custom_collection",
        )

        self.assertEqual(settings.qdrant_url, "http://localhost:7000")
        self.assertEqual(settings.collection_name, "custom_collection")
        key = settings.openrouter_api_key
        self.assertIsNotNone(key)
        if key is None:
            self.fail("openrouter_api_key should be present")

        self.assertEqual(key.get_secret_value(), "test-key")

    def test_default_cors_origins(self):
        settings = NoEnvFileSettings(openrouter_api_key=SecretStr("test-key"))
        self.assertIn("https://rag-telemetry-evals.onrender.com", settings.cors_origins)
        self.assertIn("http://localhost:5173", settings.cors_origins)

    def test_cors_origins_parsed_from_json(self):
        settings = NoEnvFileSettings(
            openrouter_api_key=SecretStr("test-key"),
            cors_origins='["https://example.com", "http://another.com"]'
        )
        self.assertEqual(settings.cors_origins, ["https://example.com", "http://another.com"])

    def test_cors_origins_parsed_from_comma_separated_string(self):
        settings = NoEnvFileSettings(
            openrouter_api_key=SecretStr("test-key"),
            cors_origins="https://example.com, http://another.com,https://third.com"
        )
        self.assertEqual(settings.cors_origins, ["https://example.com", "http://another.com", "https://third.com"])


if __name__ == "__main__":
    unittest.main()
