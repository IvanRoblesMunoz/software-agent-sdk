from enum import Enum

from pydantic import BaseModel, Field, SecretStr, field_validator


class LLMAuthStatus(str, Enum):
    """
    Status of LLM authentication configuration.

    Values describe how credentials were sourced or whether an auth profile
    is missing or unusable. This is intended for UI/diagnostics and should
    not be used as a security signal.

    - ENV: Credentials resolved from environment variables.
    - DIRECT: Credentials set directly on the LLM configuration.
    - MISSING: No credentials available from any source.
    - NOT_CONFIGURED: Auth profile name set but profile not found.
    - CONFIGURED: Auth profile found and credentials are valid.
    - CORRUPTED: Auth profile found but credentials cannot be decrypted.
    """

    ENV = "env"
    DIRECT = "direct"
    MISSING = "missing"
    NOT_CONFIGURED = "not_configured"
    CONFIGURED = "configured"
    CORRUPTED = "corrupted"


class LLMAuth(BaseModel):
    """Authentication profile for LLM providers."""

    name: str = Field(description="Unique name for this auth profile")
    credentials: dict[str, SecretStr] = Field(
        description="Provider credentials (api_key, aws_access_key_id, etc.)"
    )
    provider: str | None = Field(
        default=None, description="Optional provider hint (e.g., 'openai', 'aws')"
    )

    @field_validator("credentials", mode="before")
    @classmethod
    def coerce_credentials(cls, v: dict[str, str | SecretStr]) -> dict[str, SecretStr]:
        """Auto-coerce plain strings to SecretStr."""
        if not isinstance(v, dict):
            raise ValueError("credentials must be a dictionary")

        result = {}
        for key, value in v.items():
            if isinstance(value, SecretStr):
                result[key] = value
            elif isinstance(value, str):
                result[key] = SecretStr(value)
            else:
                raise ValueError(
                    f"Credential '{key}' must be a string or SecretStr, "
                    f"got {type(value)}"
                )
        return result

    @property
    def status(self) -> LLMAuthStatus:
        """Get the status of this auth profile."""
        if not self.credentials:
            return LLMAuthStatus.MISSING

        # Check if any credentials are None (failed decryption/corruption)
        if any(v is None for v in self.credentials.values()):
            return LLMAuthStatus.CORRUPTED

        return LLMAuthStatus.CONFIGURED

    @property
    def is_configured(self) -> bool:
        """Check if auth profile has valid credentials."""
        return self.status == LLMAuthStatus.CONFIGURED

    @property
    def is_corrupted(self) -> bool:
        """Check if auth profile has corrupted credentials."""
        return self.status == LLMAuthStatus.CORRUPTED

    @property
    def is_missing(self) -> bool:
        """Check if auth profile has no credentials."""
        return self.status == LLMAuthStatus.MISSING
