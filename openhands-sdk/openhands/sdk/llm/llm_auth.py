from enum import Enum

from pydantic import BaseModel, Field, SecretStr, field_validator

from openhands.sdk.utils.pydantic_secrets import validate_secret


class LLMAuthStatus(str, Enum):
    """
    Status of LLM authentication configuration.

    Values describe how credentials were sourced or whether an auth profile
    is missing or unusable. This is intended for UI/diagnostics and should
    not be used as a security signal.

    - DIRECT: Credentials set directly on the LLM configuration.
    - MISSING: No credentials available from any source.
    - NOT_CONFIGURED: Auth profile name set but profile not found.
    - CONFIGURED: Auth profile found and credentials are valid.
    - CORRUPTED: Auth profile found but credentials cannot be decrypted.
    """

    DIRECT = "direct"
    MISSING = "missing"
    NOT_CONFIGURED = "not_configured"
    CONFIGURED = "configured"
    CORRUPTED = "corrupted"


class LLMAuth(BaseModel):
    """Authentication profile for LLM providers."""

    name: str = Field(description="Unique name for this auth profile")
    credentials: dict[str, str | SecretStr | None] = Field(
        description="Provider credentials (api_key, aws_access_key_id, etc.)"
    )
    provider: str | None = Field(
        default=None, description="Optional provider hint (e.g., 'openai', 'aws')"
    )

    @field_validator("credentials", mode="before")
    @classmethod
    def coerce_credentials(
        cls, v: dict[str, str | SecretStr | None], info
    ) -> dict[str, SecretStr | None]:
        """Auto-coerce plain strings to SecretStr, decrypting when possible."""
        result = {}
        for key, value in v.items():
            result[key] = validate_secret(value, info)
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
    def has_valid_credentials(self) -> bool:
        """Check if auth profile has valid credentials."""
        return self.status == LLMAuthStatus.CONFIGURED
