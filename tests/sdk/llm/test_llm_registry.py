from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import SecretStr

from openhands.sdk.llm.llm import LLM
from openhands.sdk.llm.llm_registry import LLMRegistry, RegistryEvent


class TestLLMRegistry(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        # Create a registry for testing
        self.registry: LLMRegistry = LLMRegistry()

    def test_subscribe_and_notify(self):
        """Test the subscription and notification system."""
        events_received = []

        def callback(event: RegistryEvent):
            events_received.append(event)

        # Subscribe to events
        self.registry.subscribe(callback)

        # Create a mock LLM and add it to trigger notification
        mock_llm = Mock(spec=LLM)
        mock_llm.usage_id = "notify-service"

        # Mock the RegistryEvent to avoid LLM attribute access
        with patch(
            "openhands.sdk.llm.llm_registry.RegistryEvent"
        ) as mock_registry_event:
            mock_registry_event.return_value = Mock()
            self.registry.add(mock_llm)

        # Should receive notification for the newly added LLM
        self.assertEqual(len(events_received), 1)

        # Test that the subscriber is set correctly
        self.assertIsNotNone(self.registry.subscriber)

        # Test notify method directly with a mock event
        with patch.object(self.registry, "subscriber") as mock_subscriber:
            mock_event = MagicMock()
            self.registry.notify(mock_event)
            mock_subscriber.assert_called_once_with(mock_event)

    def test_registry_has_unique_id(self):
        """Test that each registry instance has a unique ID."""
        registry2 = LLMRegistry()
        self.assertNotEqual(self.registry.registry_id, registry2.registry_id)
        self.assertTrue(len(self.registry.registry_id) > 0)
        self.assertTrue(len(registry2.registry_id) > 0)


def test_llm_registry_notify_exception_handling():
    """Test LLM registry handles exceptions in subscriber notification."""

    # Create a subscriber that raises an exception
    def failing_subscriber(event):
        raise ValueError("Subscriber failed")

    registry = LLMRegistry()
    registry.subscribe(failing_subscriber)

    # Mock the logger to capture warning messages
    with patch("openhands.sdk.llm.llm_registry.logger") as mock_logger:
        # Create a mock event
        mock_event = Mock()

        # This should handle the exception and log a warning (lines 146-147)
        registry.notify(mock_event)

        # Should have logged the warning
        mock_logger.warning.assert_called_once()
        assert "Failed to emit event:" in str(mock_logger.warning.call_args)


def test_llm_registry_list_usage_ids():
    """Test LLM registry list_usage_ids method."""

    registry = LLMRegistry()

    # Create mock LLM objects
    mock_llm1 = Mock(spec=LLM)
    mock_llm1.usage_id = "service1"
    mock_llm2 = Mock(spec=LLM)
    mock_llm2.usage_id = "service2"

    # Mock the RegistryEvent to avoid LLM attribute access
    with patch("openhands.sdk.llm.llm_registry.RegistryEvent") as mock_registry_event:
        mock_registry_event.return_value = Mock()

        # Add some LLMs using the new API
        registry.add(mock_llm1)
        registry.add(mock_llm2)

        # Test list_usage_ids
        usage_ids = registry.list_usage_ids()

        assert "service1" in usage_ids
        assert "service2" in usage_ids
        assert len(usage_ids) == 2


def test_llm_registry_add_method():
    """Test the new add() method for LLMRegistry."""
    registry = LLMRegistry()

    # Create a mock LLM
    mock_llm = Mock(spec=LLM)
    mock_llm.usage_id = "test-service"
    service_id = mock_llm.usage_id

    # Mock the RegistryEvent to avoid LLM attribute access
    with patch("openhands.sdk.llm.llm_registry.RegistryEvent") as mock_registry_event:
        mock_registry_event.return_value = Mock()

        # Test adding an LLM
        registry.add(mock_llm)

        # Verify the LLM was added
        assert service_id in registry.usage_to_llm
        assert registry.usage_to_llm[service_id] is mock_llm

        # Verify RegistryEvent was called
        mock_registry_event.assert_called_once_with(llm=mock_llm)

    # Test that adding the same usage_id raises ValueError
    with unittest.TestCase().assertRaises(ValueError) as context:
        registry.add(mock_llm)

    assert "already exists in registry" in str(context.exception)


def test_llm_registry_get_method():
    """Test the new get() method for LLMRegistry."""
    registry = LLMRegistry()

    # Create a mock LLM
    mock_llm = Mock(spec=LLM)
    mock_llm.usage_id = "test-service"
    service_id = mock_llm.usage_id

    # Mock the RegistryEvent to avoid LLM attribute access
    with patch("openhands.sdk.llm.llm_registry.RegistryEvent") as mock_registry_event:
        mock_registry_event.return_value = Mock()

        # Add the LLM first
        registry.add(mock_llm)

        # Test getting the LLM
        retrieved_llm = registry.get(service_id)
        assert retrieved_llm is mock_llm

    # Test getting non-existent service raises KeyError
    with unittest.TestCase().assertRaises(KeyError) as context:
        registry.get("non-existent-service")

    assert "not found in registry" in str(context.exception)


def test_llm_registry_add_get_workflow():
    """Test the complete add/get workflow."""
    registry = LLMRegistry()

    # Create mock LLMs
    llm1 = Mock(spec=LLM)
    llm1.usage_id = "service1"
    llm2 = Mock(spec=LLM)
    llm2.usage_id = "service2"

    # Mock the RegistryEvent to avoid LLM attribute access
    with patch("openhands.sdk.llm.llm_registry.RegistryEvent") as mock_registry_event:
        mock_registry_event.return_value = Mock()

        # Add multiple LLMs
        registry.add(llm1)
        registry.add(llm2)

        # Verify we can retrieve them
        assert registry.get("service1") is llm1
        assert registry.get("service2") is llm2

        # Verify list_usage_ids works
        usage_ids = registry.list_usage_ids()
        assert "service1" in usage_ids
        assert "service2" in usage_ids
        assert len(usage_ids) == 2

        # Verify usage_id is set correctly
        assert llm1.usage_id == "service1"
        assert llm2.usage_id == "service2"


class TestLLMProfilePersistence(unittest.TestCase):
    """Tests for LLM profile save/load/delete functionality."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create temporary directory and patch profiles directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_method = LLMRegistry._get_profiles_dir

        def mock_get_profiles_dir():
            path = Path(self.temp_dir.name) / "llm_profiles"
            path.mkdir(parents=True, exist_ok=True)
            return path

        LLMRegistry._get_profiles_dir = staticmethod(mock_get_profiles_dir)
        self.sample_llm = LLM(
            model="openai/gpt-4o",
            api_key=SecretStr("sk-test-key-12345"),
            usage_id="test-agent",
            temperature=0.7,
        )

    def tearDown(self):
        """Clean up after each test."""
        LLMRegistry._get_profiles_dir = self.original_method
        self.temp_dir.cleanup()
        if "OPENHANDS_ENCRYPTION_KEY" in os.environ:
            del os.environ["OPENHANDS_ENCRYPTION_KEY"]

    def test_save_profile_with_cipher_encrypts(self):
        """Test that profiles are encrypted when OPENHANDS_ENCRYPTION_KEY is set."""
        os.environ["OPENHANDS_ENCRYPTION_KEY"] = "test-key"
        LLMRegistry.save_profile("test", self.sample_llm, expose_secrets=True)

        content = (Path(self.temp_dir.name) / "llm_profiles" / "test.json").read_text()
        # Verify the original secret is not in plaintext
        assert "sk-test-key-12345" not in content
        # Verify encryption occurred - encrypted values are base64 strings
        assert '"api_key": "' in content
        # Fernet encryption produces base64 strings starting with "gAAAAA"
        assert "gAAAAA" in content or len(content) > 500

    def test_save_profile_without_cipher_plaintext(self):
        """Test plaintext saving when no cipher and expose_secrets=True."""
        LLMRegistry.save_profile("test", self.sample_llm, expose_secrets=True)
        content = (Path(self.temp_dir.name) / "llm_profiles" / "test.json").read_text()
        assert "sk-test-key-12345" in content

    def test_save_profile_requires_cipher_or_expose_secrets(self):
        """Test that saving without cipher and expose_secrets=False raises error."""
        with pytest.raises(ValueError, match="without secrets or encryption"):
            LLMRegistry.save_profile("test", self.sample_llm, expose_secrets=False)

    def test_save_profile_override_existing(self):
        """Test override_existing flag."""
        LLMRegistry.save_profile("test", self.sample_llm, expose_secrets=True)

        with pytest.raises(FileExistsError, match="already exists"):
            LLMRegistry.save_profile("test", self.sample_llm, expose_secrets=True)

        new_llm = LLM(model="gpt-3.5", api_key=SecretStr("new"), usage_id="test")
        LLMRegistry.save_profile(
            "test", new_llm, expose_secrets=True, override_existing=True
        )
        assert LLMRegistry.load_profile("test").model == "gpt-3.5"

    def test_load_profile_with_cipher_decrypts(self):
        """Test that encrypted profiles are decrypted on load."""
        os.environ["OPENHANDS_ENCRYPTION_KEY"] = "test-key"
        LLMRegistry.save_profile("test", self.sample_llm, expose_secrets=True)

        loaded = LLMRegistry.load_profile("test")
        assert loaded.model == self.sample_llm.model
        assert loaded.api_key is not None
        assert isinstance(loaded.api_key, SecretStr)
        assert loaded.api_key.get_secret_value() == "sk-test-key-12345"
        assert loaded.usage_id == self.sample_llm.usage_id

    def test_load_profile_without_cipher_plaintext(self):
        """Test loading plaintext profiles."""
        LLMRegistry.save_profile("test", self.sample_llm, expose_secrets=True)
        loaded = LLMRegistry.load_profile("test")
        assert loaded.api_key is not None
        assert isinstance(loaded.api_key, SecretStr)
        assert loaded.api_key.get_secret_value() == "sk-test-key-12345"

    def test_load_profile_not_found(self):
        """Test loading non-existent profile raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            LLMRegistry.load_profile("missing")

    def test_load_profile_round_trip(self):
        """Test save then load preserves all attributes."""
        os.environ["OPENHANDS_ENCRYPTION_KEY"] = "test-key"
        LLMRegistry.save_profile("test", self.sample_llm, expose_secrets=True)

        loaded = LLMRegistry.load_profile("test")
        assert loaded.model == self.sample_llm.model
        assert loaded.api_key is not None
        assert isinstance(loaded.api_key, SecretStr)
        assert self.sample_llm.api_key is not None
        assert isinstance(self.sample_llm.api_key, SecretStr)
        assert (
            loaded.api_key.get_secret_value()
            == self.sample_llm.api_key.get_secret_value()
        )
        assert loaded.usage_id == self.sample_llm.usage_id
        assert loaded.temperature == self.sample_llm.temperature

    def test_delete_profile_success(self):
        """Test successful profile deletion."""
        LLMRegistry.save_profile("test", self.sample_llm, expose_secrets=True)
        profile_path = Path(self.temp_dir.name) / "llm_profiles" / "test.json"
        assert profile_path.exists()

        LLMRegistry.delete_profile("test")
        assert not profile_path.exists()

    def test_delete_profile_not_found(self):
        """Test deleting non-existent profile raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            LLMRegistry.delete_profile("missing")

    def test_list_profiles_empty(self):
        """Test listing when no profiles exist."""
        assert LLMRegistry.list_profiles() == []

    def test_list_profiles_multiple(self):
        """Test listing multiple profiles."""
        LLMRegistry.save_profile("p1", self.sample_llm, expose_secrets=True)
        LLMRegistry.save_profile("p2", self.sample_llm, expose_secrets=True)
        LLMRegistry.save_profile("p3", self.sample_llm, expose_secrets=True)

        profiles = LLMRegistry.list_profiles()
        assert len(profiles) == 3
        assert set(profiles) == {"p1", "p2", "p3"}

    def test_list_profiles_returns_names_only(self):
        """Test that list_profiles returns just names, not paths."""
        LLMRegistry.save_profile("test", self.sample_llm, expose_secrets=True)
        profiles = LLMRegistry.list_profiles()
        assert profiles == ["test"]
        assert not any("/" in p or ".json" in p for p in profiles)

    def test_full_profile_workflow(self):
        """Test complete workflow: save, list, load, delete."""
        os.environ["OPENHANDS_ENCRYPTION_KEY"] = "test-key"
        llm1 = LLM(model="gpt-4o", api_key=SecretStr("key1"), usage_id="agent")
        llm2 = LLM(model="gpt-3.5", api_key=SecretStr("key2"), usage_id="condenser")

        LLMRegistry.save_profile("agent", llm1, expose_secrets=True)
        LLMRegistry.save_profile("condenser", llm2, expose_secrets=True)

        assert len(LLMRegistry.list_profiles()) == 2
        assert LLMRegistry.load_profile("agent").model == "gpt-4o"
        assert LLMRegistry.load_profile("condenser").model == "gpt-3.5"

        LLMRegistry.delete_profile("agent")
        assert len(LLMRegistry.list_profiles()) == 1
        assert "agent" not in LLMRegistry.list_profiles()
