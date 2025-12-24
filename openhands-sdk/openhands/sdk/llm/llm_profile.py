from __future__ import annotations

from pydantic import Field, field_validator

from openhands.sdk.llm.llm import LLM
from openhands.sdk.secret import SecretSource
from openhands.sdk.utils.models import OpenHandsModel


class LLMProfile(OpenHandsModel):
    """A persistable LLM configuration that uses references for secrets."""

    name: str = Field(
        ...,
        description="The unique name of the profile (e.g., 'gpt-4o-work')",
    )

    llm: LLM = Field(
        ...,
        description="The base LLM configuration template (model, temperature, etc.)",
    )

    secrets: dict[str, SecretSource] = Field(
        default_factory=dict,
        description="Mapping of LLM secret field names to their sources",
    )

    @field_validator("secrets")
    @staticmethod
    def _validate_secret_keys(
        v: dict[str, SecretSource],
    ) -> dict[str, SecretSource]:
        """Ensures that the secret keys correspond to valid LLM fields."""
        # Get all valid fields from the LLM class
        allowed_fields = set(LLM.model_fields.keys())

        for key in v:
            if key not in allowed_fields:
                raise ValueError(
                    f"'{key}' is not a valid configuration field in the LLM class. "
                    f"Allowed secret fields are often: api_key, aws_access_key_id, ..."
                )
        return v

    def resolve_llm(self) -> LLM:
        """
        Resolves all SecretSource references (env vars, URLs) and
        returns a fully-functional LLM instance ready for use.
        """
        updates = {}
        for field, source in self.secrets.items():
            val = source.get_value()
            if val:
                updates[field] = val

        # 1. Start with the template settings
        config = self.llm.model_dump()

        # 2. Layer on the resolved secrets
        config.update(updates)

        # 3. Create the live instance (runs all Pydantic validators)
        return LLM(**config)
