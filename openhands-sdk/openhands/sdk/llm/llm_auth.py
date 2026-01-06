from enum import Enum

from pydantic import BaseModel, Field, SecretStr, field_serializer, field_validator

from openhands.sdk.utils.pydantic_secrets import serialize_secret, validate_secret


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
    - UNREADABLE: Auth profile found but credentials cannot be decrypted.
    """

    DIRECT = "direct"
    MISSING = "missing"
    NOT_CONFIGURED = "not_configured"
    CONFIGURED = "configured"
    UNREADABLE = "unreadable"


class LLMAuth(BaseModel):
    """Authentication profile for LLM providers."""

    name: str = Field(description="Unique name for this auth profile")
    credentials: dict[str, str | SecretStr | None] = Field(
        description="Provider credentials (api_key, aws_access_key_id, etc.)"
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

    @field_serializer("credentials", when_used="always")
    def _serialize_credentials(self, v: dict[str, SecretStr | None], info):
        result: dict[str, str | SecretStr | None] = {}
        for key, value in v.items():
            result[key] = serialize_secret(value, info)
        return result

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate auth profile names using registry rules."""
        from openhands.sdk.llm.llm_registry import LLMRegistry

        return LLMRegistry._validate_profile_name(v)

    @property
    def status(self) -> LLMAuthStatus:
        """Get the status of this auth profile."""
        if not self.credentials:
            return LLMAuthStatus.MISSING

        # Check if any credentials are None (failed decryption/corruption)
        if any(v is None for v in self.credentials.values()):
            return LLMAuthStatus.UNREADABLE

        return LLMAuthStatus.CONFIGURED

    @property
    def has_valid_credentials(self) -> bool:
        """Check if auth profile has valid credentials."""
        return self.status == LLMAuthStatus.CONFIGURED
