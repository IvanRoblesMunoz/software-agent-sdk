from __future__ import annotations

import pytest
from pydantic import SecretStr

from openhands.sdk.llm.llm_auth import LLMAuth, LLMAuthStatus


@pytest.fixture(autouse=True)
def _clear_llm_auth_env(monkeypatch):
    monkeypatch.delenv("OPENHANDS_ENCRYPTION_KEY", raising=False)


@pytest.mark.parametrize(
    "name,credentials,expected_status,expected_has_valid_credentials",
    [
        ("missing", {}, LLMAuthStatus.MISSING, False),
        ("corrupted", {"api_key": None}, LLMAuthStatus.CORRUPTED, False),
        ("configured", {"api_key": SecretStr("sk")}, LLMAuthStatus.CONFIGURED, True),
    ],
)
def test_llm_auth_status_and_properties(
    name, credentials, expected_status, expected_has_valid_credentials
):
    auth = LLMAuth(name=name, credentials=credentials)
    assert auth.status == expected_status
    assert auth.has_valid_credentials is expected_has_valid_credentials


def test_llm_auth_coerces_string_credentials():
    auth = LLMAuth(name="coerce", credentials={"api_key": "sk-test"})
    assert auth.credentials["api_key"] is not None
    assert isinstance(auth.credentials["api_key"], SecretStr)
    assert auth.credentials["api_key"].get_secret_value() == "sk-test"
