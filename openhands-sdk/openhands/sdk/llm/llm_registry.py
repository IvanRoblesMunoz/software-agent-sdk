import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from openhands.sdk.llm.llm import LLM
from openhands.sdk.llm.llm_profile import LLMProfile
from openhands.sdk.logger import get_logger
from openhands.sdk.secret import StaticSecret
from openhands.sdk.utils.pydantic_secrets import Cipher


logger = get_logger(__name__)


class RegistryEvent(BaseModel):
    llm: LLM

    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True,
    )


class LLMRegistry:
    """A minimal LLM registry for managing LLM instances by usage ID.

    This registry provides a simple way to manage multiple LLM instances,
    avoiding the need to recreate LLMs with the same configuration.
    """

    registry_id: str
    retry_listener: Callable[[int, int], None] | None

    def __init__(
        self,
        retry_listener: Callable[[int, int], None] | None = None,
    ):
        """Initialize the LLM registry.

        Args:
            retry_listener: Optional callback for retry events.
        """
        self.registry_id = str(uuid4())
        self.retry_listener = retry_listener
        self._usage_to_llm: dict[str, LLM] = {}
        self.subscriber: Callable[[RegistryEvent], None] | None = None

    def subscribe(self, callback: Callable[[RegistryEvent], None]) -> None:
        """Subscribe to registry events.

        Args:
            callback: Function to call when LLMs are created or updated.
        """
        self.subscriber = callback

    def notify(self, event: RegistryEvent) -> None:
        """Notify subscribers of registry events.

        Args:
            event: The registry event to notify about.
        """
        if self.subscriber:
            try:
                self.subscriber(event)
            except Exception as e:
                logger.warning(f"Failed to emit event: {e}")

    @property
    def usage_to_llm(self) -> dict[str, LLM]:
        """Access the internal usage-ID-to-LLM mapping."""

        return self._usage_to_llm

    def add(self, llm: LLM) -> None:
        """Add an LLM instance to the registry.

        Args:
            llm: The LLM instance to register.

        Raises:
            ValueError: If llm.usage_id already exists in the registry.
        """
        usage_id = llm.usage_id
        if usage_id in self._usage_to_llm:
            message = (
                f"Usage ID '{usage_id}' already exists in registry. "
                "Use a different usage_id on the LLM or "
                "call get() to retrieve the existing LLM."
            )
            raise ValueError(message)

        self._usage_to_llm[usage_id] = llm
        self.notify(RegistryEvent(llm=llm))
        logger.debug(
            f"[LLM registry {self.registry_id}]: Added LLM for usage {usage_id}"
        )

    def get(self, usage_id: str) -> LLM:
        """Get an LLM instance from the registry.

        Args:
            usage_id: Unique identifier for the LLM usage slot.

        Returns:
            The LLM instance.

        Raises:
            KeyError: If usage_id is not found in the registry.
        """
        if usage_id not in self._usage_to_llm:
            raise KeyError(
                f"Usage ID '{usage_id}' not found in registry. "
                "Use add() to register an LLM first."
            )

        logger.info(
            f"[LLM registry {self.registry_id}]: Retrieved LLM for usage {usage_id}"
        )
        return self._usage_to_llm[usage_id]

    def list_usage_ids(self) -> list[str]:
        """List all registered usage IDs."""

        return list(self._usage_to_llm.keys())

    @staticmethod
    def _get_profiles_dir() -> Path:
        """Get the standard directory for LLM profiles."""
        path = Path.home() / ".openhands" / "profiles"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def list_profiles(cls) -> list[str]:
        """List the names of all available LLM profiles."""
        profiles_dir = cls._get_profiles_dir()
        return [f.stem for f in profiles_dir.glob("*.json")]

    def save_profile(self, profile: LLMProfile, allow_unsafe: bool = False) -> None:
        """Save an LLM profile to disk.

        Args:
            profile: The profile to save.
            allow_unsafe: If True, allows saving profiles with raw StaticSecret
                          keys in plain text (if no encryption key is set).
                          Defaults to False.
        """
        # Safety check: Ensure no raw secrets are being saved unless explicitly allowed
        if not allow_unsafe:
            for field_name, source in profile.secrets.items():
                if isinstance(source, StaticSecret):
                    # Check if encryption is enabled
                    if not os.environ.get("OPENHANDS_ENCRYPTION_KEY"):
                        raise ValueError(
                            f"Safety Gate: Secret '{field_name}' in profile "
                            f"'{profile.name}' is a StaticSecret, but no "
                            "OPENHANDS_ENCRYPTION_KEY is set. "
                            "Refusing to save plain-text secrets to disk. "
                            "Use EnvSecret instead, or set "
                            "OPENHANDS_ENCRYPTION_KEY."
                        )

        profile_path = self._get_profiles_dir() / f"{profile.name}.json"

        # Setup encryption context if key exists
        enc_key = os.environ.get("OPENHANDS_ENCRYPTION_KEY")

        context: dict[str, Any] = {"cipher": Cipher(enc_key)} if enc_key else {}

        # We always want to expose secrets to the cipher or for raw save if allowed
        context["expose_secrets"] = True

        serialized = profile.model_dump_json(context=context)

        with open(profile_path, "w") as f:
            f.write(serialized)

        logger.info(f"Saved LLM profile '{profile.name}' to {profile_path}")

    def load_profile(self, name: str) -> LLMProfile:
        """Load an LLM profile from disk.

        Args:
            name: The name of the profile.

        Returns:
            The loaded LLMProfile.
        """
        profile_path = self._get_profiles_dir() / f"{name}.json"
        if not profile_path.exists():
            raise FileNotFoundError(f"Profile '{name}' not found at {profile_path}")

        with open(profile_path) as f:
            json_data = f.read()

        # Setup decryption context if key exists
        enc_key = os.environ.get("OPENHANDS_ENCRYPTION_KEY")
        context = {"cipher": Cipher(enc_key)} if enc_key else {}

        profile = LLMProfile.model_validate_json(json_data, context=context)
        return profile
