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


if __name__ == "__main__":
    unittest.main()
