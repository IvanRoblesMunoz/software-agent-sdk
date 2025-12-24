from __future__ import annotations

from pydantic import Field

from openhands.sdk.llm.llm import LLM
from openhands.sdk.secret import SecretSource
from openhands.sdk.utils.models import OpenHandsModel


class LLMProfile(OpenHandsModel):
    """A persistable LLM configuration that uses references for secrets."""

    name: str  # The unique name of the profile (e.g., 'gpt-4o-work')

    # The actual LLM configuration (model, temperature, etc.)
    # The secret fields in this object will typically be None or redacted.
    llm: LLM

    # A mapping of secret field names to their sources
    # e.g., {"api_key": EnvSecret(env_var="OPENAI_API_KEY")}
    secrets: dict[str, SecretSource] = Field(default_factory=dict)

    def hydrate(self) -> LLM:
        """Returns a fully-functional LLM by resolving the secret sources."""
        updates = {}
        for field, source in self.secrets.items():
            val = source.get_value()
            if val:
                updates[field] = val

        # We use model_dump + update + LLM() to ensure validators
        # (like SecretStr promotion) run
        config = self.llm.model_dump()
        config.update(updates)
        return LLM(**config)
