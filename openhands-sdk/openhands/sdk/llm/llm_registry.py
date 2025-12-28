import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from openhands.sdk.llm.llm import LLM
from openhands.sdk.logger import get_logger
from openhands.sdk.utils.pydantic_secrets import Cipher


logger = get_logger(__name__)

_VALID_LLM_PROFILE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


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
    profile_name: str | None

    def __init__(
        self,
        retry_listener: Callable[[int, int], None] | None = None,
        profile_name: str | None = None,
    ):
        """Initialize the LLM registry.

        Args:
            retry_listener: Optional callback for retry events.
            profile_name: Optional name for this registry profile.
                         Required when saving registry profiles.
        """
        self.registry_id = str(uuid4())
        self.retry_listener = retry_listener
        self.profile_name = profile_name
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

    def add_llms_from_profiles(self, usage_to_profile: dict[str, str]) -> None:
        """Load and add multiple LLM profiles to the registry.

        Args:
            usage_to_profile: Mapping of usage IDs to profile names.
                             Keys are usage IDs, values are profile names.

        Example:
            >>> registry = LLMRegistry()
            >>> registry.add_llms_from_profiles({
            ...     "agent": "claude-sonnet",
            ...     "title-gen": "gpt-4o",
            ...     "code-gen": "o3-mini",
            ... })
        """
        for usage_id, profile_name in usage_to_profile.items():
            llm = self.load_llm_profile(profile_name, usage_id=usage_id)
            self.add(llm)

    @staticmethod
    def _get_profiles_dir() -> Path:
        """Get the standard directory for LLM profiles."""
        path = Path.home() / ".openhands" / "llm_profiles"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def _get_profile_path(cls, name: str) -> Path:
        return cls._get_profiles_dir() / f"{name}.json"

    @staticmethod
    def _validate_profile_name(name: str) -> str:
        """Ensure profile name is safe for filesystem use."""
        if not name or name in {".", ".."}:
            raise ValueError("Profile name cannot be '.', '..', or empty")

        if Path(name).name != name:
            raise ValueError("Profile name cannot contain path separators")

        if not _VALID_LLM_PROFILE_NAME_PATTERN.fullmatch(name):
            raise ValueError(
                "Profile name must contain only alphanumerics, dots, "
                "underscores, or hyphens"
            )

        return name

    @staticmethod
    def _get_cipher_context() -> dict[str, Any]:
        enc_key = os.environ.get("OPENHANDS_ENCRYPTION_KEY")
        return {"cipher": Cipher(enc_key)} if enc_key else {}

    @classmethod
    def list_llm_profiles(cls) -> list[str]:
        """List the names of all available LLM profiles."""
        return [f.stem for f in cls._get_profiles_dir().glob("*.json")]

    @classmethod
    def save_llm_profile(
        cls,
        llm: LLM,
        override_existing: bool = False,
    ) -> None:
        """
        Save an LLM instance as a named profile.

        The profile name is taken from llm.profile_name, which must be set.
        Secrets are automatically encrypted if OPENHANDS_ENCRYPTION_KEY is set.

        Args:
            llm: The LLM instance to save. Must have profile_name set.
            override_existing: If True, overwrite existing profile.

        Raises:
            ValueError: If llm.profile_name is not set.
            FileExistsError: If profile already exists and override_existing is False.
        """
        if not llm.profile_name:
            raise ValueError(
                "LLM must have profile_name set before saving. "
                "Set llm.profile_name to the desired profile name."
            )

        name = llm.profile_name
        cls._validate_profile_name(name)
        profile_path = cls._get_profile_path(name)
        if profile_path.exists() and not override_existing:
            raise FileExistsError(
                f"Profile '{name}' already exists. "
                "Use override_existing=True to overwrite."
            )

        context = cls._get_cipher_context()
        has_cipher = "cipher" in context

        if not has_cipher:
            logger.warning(
                f"Saving profile '{name}' without encryption. "
                "Secrets will be redacted. "
                "Set OPENHANDS_ENCRYPTION_KEY environment variable to encrypt secrets."
            )

        with open(profile_path, "w") as f:
            json.dump(llm.model_dump(context=context, mode="json"), f, indent=2)

        status = "encrypted" if has_cipher else "plaintext"
        logger.info(f"Saved LLM profile '{name}' ({status}) to {profile_path}")

    @classmethod
    def load_llm_profile(cls, name: str, usage_id: str | None = None) -> LLM:
        """Load an LLM profile from disk.

        Args:
            name: Name of the profile to load.
            usage_id: Optional usage_id to set on the loaded LLM.
                     If not provided, keeps the default ("default").

        Returns:
            The loaded LLM instance with profile_name set to name.
        """
        profile_path = cls._get_profile_path(name)
        if not profile_path.exists():
            raise FileNotFoundError(f"Profile '{name}' not found.")

        with open(profile_path) as f:
            data = json.load(f)

        llm = LLM.model_validate(data, context=cls._get_cipher_context())

        # Set profile_name to track where this LLM came from
        llm.profile_name = name

        # Optionally override usage_id
        if usage_id is not None:
            llm.usage_id = usage_id

        logger.info(
            f"Loaded LLM profile '{name}'"
            + (f" with usage_id='{usage_id}'" if usage_id else "")
        )
        return llm

    @classmethod
    def delete_llm_profile(cls, name: str) -> None:
        """Delete an LLM profile from disk."""
        profile_path = cls._get_profile_path(name)
        if not profile_path.exists():
            raise FileNotFoundError(f"Profile '{name}' not found.")

        profile_path.unlink()
        logger.info(f"Deleted LLM profile '{name}'")

    # =========================================================================
    # Registry profile helpers
    # =========================================================================
    @staticmethod
    def _get_registry_profiles_dir() -> Path:
        """Get the standard directory for registry profiles."""
        path = Path.home() / ".openhands" / "registry_profiles"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def _get_registry_profile_path(cls, name: str) -> Path:
        return cls._get_registry_profiles_dir() / f"{name}.json"

    @classmethod
    def list_registry_profiles(cls) -> list[str]:
        """List the names of all available registry profiles."""
        return [f.stem for f in cls._get_registry_profiles_dir().glob("*.json")]

    @classmethod
    def save_registry_profile(
        cls,
        registry: "LLMRegistry",
        override_existing: bool = False,
    ) -> None:
        """
        Save a registry with all its LLMs as a named profile.

        The profile name is taken from registry.profile_name, which must be set.
        Secrets are automatically encrypted if OPENHANDS_ENCRYPTION_KEY is set.

        Args:
            registry: The LLMRegistry instance to save. Must have profile_name set.
            override_existing: If True, overwrite existing profile.

        Raises:
            ValueError: If registry.profile_name is not set.
            FileExistsError: If profile already exists and override_existing is False.
        """
        if not registry.profile_name:
            raise ValueError(
                "Registry must have profile_name set before saving. "
                "Set registry.profile_name to the desired profile name."
            )

        name = registry.profile_name
        cls._validate_profile_name(name)
        profile_path = cls._get_registry_profile_path(name)
        if profile_path.exists() and not override_existing:
            raise FileExistsError(
                f"Registry profile '{name}' already exists. "
                "Use override_existing=True to overwrite."
            )

        context = cls._get_cipher_context()
        has_cipher = "cipher" in context

        if not has_cipher:
            logger.warning(
                f"Saving registry profile '{name}' without encryption. "
                "Secrets will be redacted. "
                "Set OPENHANDS_ENCRYPTION_KEY environment variable to encrypt secrets."
            )

        # Serialize all LLMs in the registry
        llms_data = {}
        for usage_id, llm in registry._usage_to_llm.items():
            llms_data[usage_id] = llm.model_dump(context=context, mode="json")

        registry_data = {"llms": llms_data}

        with open(profile_path, "w") as f:
            json.dump(registry_data, f, indent=2)

        status = "encrypted" if has_cipher else "plaintext"
        llm_count = len(llms_data)
        logger.info(
            f"Saved registry profile '{name}' with {llm_count} LLM(s) "
            f"({status}) to {profile_path}"
        )

    @classmethod
    def load_registry_profile(cls, name: str) -> "LLMRegistry":
        """Load a registry profile from disk.

        Args:
            name: Name of the registry profile to load.

        Returns:
            A new LLMRegistry instance populated with all LLMs from the profile.
            The registry's profile_name will be set to name.
        """
        profile_path = cls._get_registry_profile_path(name)
        if not profile_path.exists():
            raise FileNotFoundError(f"Registry profile '{name}' not found.")

        with open(profile_path) as f:
            data = json.load(f)

        # Create new registry with the profile name
        registry = cls(profile_name=name)

        # Load and add all LLMs
        cipher_context = cls._get_cipher_context()
        llms_data = data.get("llms", {})

        for usage_id, llm_data in llms_data.items():
            llm = LLM.model_validate(llm_data, context=cipher_context)
            # Ensure usage_id matches the key
            llm.usage_id = usage_id
            registry.add(llm)

        logger.info(f"Loaded registry profile '{name}' with {len(llms_data)} LLM(s)")
        return registry

    @classmethod
    def delete_registry_profile(cls, name: str) -> None:
        """Delete a registry profile from disk."""
        profile_path = cls._get_registry_profile_path(name)
        if not profile_path.exists():
            raise FileNotFoundError(f"Registry profile '{name}' not found.")

        profile_path.unlink()
        logger.info(f"Deleted registry profile '{name}'")
