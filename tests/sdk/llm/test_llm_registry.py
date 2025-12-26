from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import ResponseOutputText
from pydantic import SecretStr

from openhands.sdk.llm.llm import LLM
from openhands.sdk.llm.llm_auth import LLMAuth, LLMAuthStatus
from openhands.sdk.llm.llm_registry import LLMRegistry, RegistryEvent
from openhands.sdk.llm.message import Message, TextContent
from tests.conftest import create_mock_litellm_response


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


class TestLLMProfilePersistence:
    """Tests for LLM profile save/load/delete functionality."""

    @pytest.fixture(autouse=True)
    def setup_profile_persistence(self):
        """Set up test environment before each test."""
        # Create temporary directory and patch profiles directory
        temp_dir = tempfile.TemporaryDirectory()
        original_llm_profiles_method = LLMRegistry._get_profiles_dir
        original_registry_profiles_method = LLMRegistry._get_registry_profiles_dir
        original_key = os.environ.get("OPENHANDS_ENCRYPTION_KEY")

        def mock_get_profiles_dir():
            path = Path(temp_dir.name) / "llm_profiles"
            path.mkdir(parents=True, exist_ok=True)
            return path

        def mock_get_registry_profiles_dir():
            path = Path(temp_dir.name) / "registry_profiles"
            path.mkdir(parents=True, exist_ok=True)
            return path

        LLMRegistry._get_profiles_dir = staticmethod(mock_get_profiles_dir)
        LLMRegistry._get_registry_profiles_dir = staticmethod(
            mock_get_registry_profiles_dir
        )

        sample_llm = LLM(
            model="openai/gpt-4o",
            api_key=SecretStr("sk-test-key-12345"),
            usage_id="test-agent",
            temperature=0.7,
        )

        # Store in instance for test methods to access
        self.temp_dir = temp_dir
        self.sample_llm = sample_llm

        os.environ.pop("OPENHANDS_ENCRYPTION_KEY", None)

        yield

        # Cleanup
        LLMRegistry._get_profiles_dir = original_llm_profiles_method
        LLMRegistry._get_registry_profiles_dir = original_registry_profiles_method
        temp_dir.cleanup()
        if original_key is None:
            os.environ.pop("OPENHANDS_ENCRYPTION_KEY", None)
        else:
            os.environ["OPENHANDS_ENCRYPTION_KEY"] = original_key

    def test_save_llm_profile_with_cipher_encrypts(self):
        """Test that profiles are encrypted when OPENHANDS_ENCRYPTION_KEY is set."""
        os.environ["OPENHANDS_ENCRYPTION_KEY"] = "test-key"
        LLMRegistry.save_llm_profile("test", self.sample_llm)

        content = (Path(self.temp_dir.name) / "llm_profiles" / "test.json").read_text()
        # Verify the original secret is not in plaintext
        assert "sk-test-key-12345" not in content
        # Verify encryption occurred - encrypted values are base64 strings
        assert '"api_key": "' in content
        # Fernet encryption produces base64 strings starting with "gAAAAA"
        assert "gAAAAA" in content or len(content) > 500

    def test_save_llm_profile_without_cipher_redacted(self):
        """
        Test saving and loading without cipher. Secrets redacted and warning logged.
        """
        with patch("openhands.sdk.llm.llm_registry.logger") as mock_logger:
            LLMRegistry.save_llm_profile("test", self.sample_llm)
            mock_logger.warning.assert_called_once()
            assert "without encryption" in str(mock_logger.warning.call_args)

        # Verify secrets are redacted in saved file
        content = (Path(self.temp_dir.name) / "llm_profiles" / "test.json").read_text()
        assert "sk-test-key-12345" not in content

        # Verify loading works (secrets will be redacted)
        loaded = LLMRegistry.load_llm_profile("test")
        assert loaded.model == self.sample_llm.model

    def test_save_llm_profile_override_existing(self):
        """Test override_existing flag."""
        LLMRegistry.save_llm_profile("test", self.sample_llm)

        with pytest.raises(FileExistsError, match="already exists"):
            LLMRegistry.save_llm_profile("test", self.sample_llm)

        new_llm = LLM(
            model="gpt-3.5",
            api_key=SecretStr("new"),
            usage_id="test",
        )
        LLMRegistry.save_llm_profile("test", new_llm, override_existing=True)
        assert LLMRegistry.load_llm_profile("test").model == "gpt-3.5"

    def test_load_llm_profile_not_found(self):
        """Test loading non-existent profile raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            LLMRegistry.load_llm_profile("missing")

    def test_load_llm_profile_round_trip(self):
        """Test save then load preserves all attributes."""
        os.environ["OPENHANDS_ENCRYPTION_KEY"] = "test-key"
        LLMRegistry.save_llm_profile("test", self.sample_llm)

        loaded = LLMRegistry.load_llm_profile("test")
        assert loaded.model == self.sample_llm.model
        assert loaded.api_key is not None
        assert isinstance(loaded.api_key, SecretStr)
        assert self.sample_llm.api_key is not None
        assert isinstance(self.sample_llm.api_key, SecretStr)
        assert (
            loaded.api_key.get_secret_value()
            == self.sample_llm.api_key.get_secret_value()
        )
        assert loaded.temperature == self.sample_llm.temperature

    def test_delete_llm_profile_success(self):
        """Test successful profile deletion."""
        LLMRegistry.save_llm_profile("test", self.sample_llm)
        profile_path = Path(self.temp_dir.name) / "llm_profiles" / "test.json"
        assert profile_path.exists()

        LLMRegistry.delete_llm_profile("test")
        assert not profile_path.exists()

    def test_delete_llm_profile_not_found(self):
        """Test deleting non-existent profile raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            LLMRegistry.delete_llm_profile("missing")

    def test_list_llm_profiles_empty(self):
        """Test listing when no profiles exist."""
        assert LLMRegistry.list_llm_profiles() == []

    def test_list_llm_profiles_multiple(self):
        """Test listing multiple profiles."""
        LLMRegistry.save_llm_profile("p1", self.sample_llm)
        LLMRegistry.save_llm_profile("p2", self.sample_llm)
        LLMRegistry.save_llm_profile("p3", self.sample_llm)

        profiles = LLMRegistry.list_llm_profiles()
        assert len(profiles) == 3
        assert set(profiles) == {"p1", "p2", "p3"}

    def test_list_llm_profiles_returns_names_only(self):
        """Test that list_llm_profiles returns just names, not paths."""
        LLMRegistry.save_llm_profile("test", self.sample_llm)
        profiles = LLMRegistry.list_llm_profiles()
        assert profiles == ["test"]
        assert not any("/" in p or ".json" in p for p in profiles)

    def test_full_profile_workflow(self):
        """Test complete workflow: save, list, load, delete."""
        os.environ["OPENHANDS_ENCRYPTION_KEY"] = "test-key"
        llm1 = LLM(
            model="gpt-4o",
            api_key=SecretStr("key1"),
            usage_id="agent",
        )
        llm2 = LLM(
            model="gpt-3.5",
            api_key=SecretStr("key2"),
            usage_id="condenser",
        )

        LLMRegistry.save_llm_profile("agent", llm1)
        LLMRegistry.save_llm_profile("condenser", llm2)

        assert len(LLMRegistry.list_llm_profiles()) == 2
        assert LLMRegistry.load_llm_profile("agent").model == "gpt-4o"
        assert LLMRegistry.load_llm_profile("condenser").model == "gpt-3.5"

        LLMRegistry.delete_llm_profile("agent")
        assert len(LLMRegistry.list_llm_profiles()) == 1
        assert "agent" not in LLMRegistry.list_llm_profiles()

    @pytest.mark.parametrize(
        "name,should_raise,error_match",
        [
            # Invalid names
            ("", True, "cannot be"),
            (".", True, "cannot be"),
            ("..", True, "cannot be"),
            ("test/profile", True, "path separators"),
            ("../test", True, "path separators"),
            ("test@profile", True, "alphanumerics"),
            ("test profile", True, "alphanumerics"),
            # Valid names
            ("test", False, None),
            ("test-profile", False, None),
            ("test_profile", False, None),
            ("test.profile", False, None),
            ("test123", False, None),
            ("Test123_Profile-Name", False, None),
        ],
    )
    def test_validate_profile_name(self, name, should_raise, error_match):
        """Test profile name validation."""
        if should_raise:
            with pytest.raises(ValueError, match=error_match):
                LLMRegistry.save_llm_profile(name, self.sample_llm)
        else:
            LLMRegistry.save_llm_profile(name, self.sample_llm, override_existing=True)
            assert name in LLMRegistry.list_llm_profiles()

    def test_load_llm_profile_with_usage_id(self):
        """Test loading profile with custom usage_id."""
        os.environ["OPENHANDS_ENCRYPTION_KEY"] = "test-key"
        LLMRegistry.save_llm_profile("test", self.sample_llm)

        loaded = LLMRegistry.load_llm_profile("test", usage_id="custom-usage")
        assert loaded.usage_id == "custom-usage"

    def test_save_registry_profile(self):
        """Test saving a registry profile."""
        os.environ["OPENHANDS_ENCRYPTION_KEY"] = "test-key"
        registry = LLMRegistry()

        llm1 = LLM(
            model="gpt-4o",
            api_key=SecretStr("key1"),
            usage_id="agent",
        )
        llm2 = LLM(
            model="gpt-3.5",
            api_key=SecretStr("key2"),
            usage_id="title-gen",
        )

        registry.add(llm1)
        registry.add(llm2)

        registry.save_registry_profile("test-registry")

        # Verify file exists
        registry_path = (
            Path(self.temp_dir.name) / "registry_profiles" / "test-registry.json"
        )
        assert registry_path.exists()

    def test_load_registry_profile(self):
        """Test loading a registry profile."""
        os.environ["OPENHANDS_ENCRYPTION_KEY"] = "test-key"
        registry = LLMRegistry()

        llm1 = LLM(model="gpt-4o", api_key=SecretStr("key1"), usage_id="agent")
        llm2 = LLM(model="gpt-3.5", api_key=SecretStr("key2"), usage_id="title-gen")

        registry.add(llm1)
        registry.add(llm2)

        registry.save_registry_profile("test-registry")

        # Load it back
        loaded_registry = LLMRegistry.load_registry_profile("test-registry")
        assert len(loaded_registry.list_usage_ids()) == 2
        assert "agent" in loaded_registry.list_usage_ids()
        assert "title-gen" in loaded_registry.list_usage_ids()

        loaded_llm1 = loaded_registry.get("agent")
        assert loaded_llm1.model == "gpt-4o"
        assert loaded_llm1.usage_id == "agent"

    def test_save_registry_profile_validates_name(self):
        """Test that save_registry_profile validates the profile name."""
        registry = LLMRegistry()
        registry.add(LLM(model="gpt-4o", api_key=SecretStr("key"), usage_id="test"))

        # Test with invalid name
        with pytest.raises(ValueError, match="alphanumerics"):
            registry.save_registry_profile("invalid name")

    def test_list_registry_profiles(self):
        """Test listing registry profiles."""
        os.environ["OPENHANDS_ENCRYPTION_KEY"] = "test-key"

        reg1 = LLMRegistry()
        reg1.add(LLM(model="gpt-4o", api_key=SecretStr("key"), usage_id="test"))
        reg1.save_registry_profile("reg1")

        reg2 = LLMRegistry()
        reg2.add(LLM(model="gpt-3.5", api_key=SecretStr("key"), usage_id="test"))
        reg2.save_registry_profile("reg2")

        profiles = LLMRegistry.list_registry_profiles()
        assert len(profiles) == 2
        assert set(profiles) == {"reg1", "reg2"}

    def test_delete_registry_profile(self):
        """Test deleting a registry profile."""
        os.environ["OPENHANDS_ENCRYPTION_KEY"] = "test-key"
        registry = LLMRegistry()
        registry.add(LLM(model="gpt-4o", api_key=SecretStr("key"), usage_id="test"))
        registry.save_registry_profile("test-registry")

        registry_path = (
            Path(self.temp_dir.name) / "registry_profiles" / "test-registry.json"
        )
        assert registry_path.exists()

        LLMRegistry.delete_registry_profile("test-registry")
        assert not registry_path.exists()

    def test_add_llms_from_profiles(self):
        """Test adding multiple LLMs from profiles to a registry."""
        os.environ["OPENHANDS_ENCRYPTION_KEY"] = "test-key"

        # Create and save some LLM profiles
        llm1 = LLM(model="gpt-4o", api_key=SecretStr("key1"))
        llm2 = LLM(model="gpt-3.5", api_key=SecretStr("key2"))
        llm3 = LLM(model="claude-3", api_key=SecretStr("key3"))

        LLMRegistry.save_llm_profile("profile1", llm1)
        LLMRegistry.save_llm_profile("profile2", llm2)
        LLMRegistry.save_llm_profile("profile3", llm3)

        # Create registry and bulk load
        registry = LLMRegistry()
        registry.add_llms_from_profiles(
            {
                "agent": "profile1",
                "title-gen": "profile2",
                "code-gen": "profile3",
            }
        )

        # Verify all were loaded with correct usage_ids
        assert len(registry.list_usage_ids()) == 3
        assert set(registry.list_usage_ids()) == {"agent", "title-gen", "code-gen"}

        # Verify LLMs have correct models and usage_ids
        assert registry.get("agent").model == "gpt-4o"
        assert registry.get("agent").usage_id == "agent"

        assert registry.get("title-gen").model == "gpt-3.5"
        assert registry.get("title-gen").usage_id == "title-gen"

        assert registry.get("code-gen").model == "claude-3"
        assert registry.get("code-gen").usage_id == "code-gen"


class TestLLMAuthProfilePersistence:
    """Tests for auth profile save/load/delete functionality."""

    @pytest.fixture(autouse=True)
    def setup_auth_profile_persistence(self):
        """Set up test environment before each test."""
        temp_dir = tempfile.TemporaryDirectory()
        original_method = LLMRegistry._get_auth_profiles_dir
        original_key = os.environ.get("OPENHANDS_ENCRYPTION_KEY")

        def mock_get_auth_profiles_dir():
            path = Path(temp_dir.name) / "auth_profiles"
            path.mkdir(parents=True, exist_ok=True)
            return path

        LLMRegistry._get_auth_profiles_dir = staticmethod(mock_get_auth_profiles_dir)
        sample_auth = LLMAuth(
            name="test",
            credentials={"api_key": SecretStr("sk-test-auth-12345")},
        )

        self.temp_dir = temp_dir
        self.sample_auth = sample_auth

        os.environ.pop("OPENHANDS_ENCRYPTION_KEY", None)

        yield

        LLMRegistry._get_auth_profiles_dir = original_method
        temp_dir.cleanup()
        if original_key is None:
            os.environ.pop("OPENHANDS_ENCRYPTION_KEY", None)
        else:
            os.environ["OPENHANDS_ENCRYPTION_KEY"] = original_key

    def test_save_auth_profile_with_cipher_redacts_secret(self):
        """Test that auth profiles do not store secrets in plaintext."""
        os.environ["OPENHANDS_ENCRYPTION_KEY"] = "test-key"
        LLMRegistry.save_auth_profile(self.sample_auth)

        content = (Path(self.temp_dir.name) / "auth_profiles" / "test.json").read_text()
        assert "sk-test-auth-12345" not in content
        assert '"api_key":' in content

    def test_save_auth_profile_without_cipher_warns_and_redacts(self):
        """Test saving without cipher warns and secrets are redacted."""
        with patch("openhands.sdk.llm.llm_registry.logger") as mock_logger:
            LLMRegistry.save_auth_profile(self.sample_auth)
            mock_logger.warning.assert_called_once()
            assert "without encryption" in str(mock_logger.warning.call_args)

        content = (Path(self.temp_dir.name) / "auth_profiles" / "test.json").read_text()
        assert "sk-test-auth-12345" not in content

        loaded = LLMRegistry.load_auth_profile("test")
        assert loaded.name == self.sample_auth.name
        assert loaded.status == LLMAuthStatus.UNREADABLE

    def test_save_auth_profile_override_existing(self):
        """Test override_existing flag."""
        LLMRegistry.save_auth_profile(self.sample_auth)

        with pytest.raises(FileExistsError, match="already exists"):
            LLMRegistry.save_auth_profile(self.sample_auth)

        new_auth = LLMAuth(
            name="test",
            credentials={"api_key": SecretStr("new")},
        )
        os.environ["OPENHANDS_ENCRYPTION_KEY"] = "test-key"
        LLMRegistry.save_auth_profile(new_auth, override_existing=True)
        loaded = LLMRegistry.load_auth_profile("test")
        assert loaded.credentials["api_key"] is not None

    def test_load_auth_profile_not_found(self):
        """Test loading non-existent profile raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            LLMRegistry.load_auth_profile("missing")

    def test_delete_auth_profile_success(self):
        """Test successful profile deletion."""
        LLMRegistry.save_auth_profile(self.sample_auth)
        profile_path = Path(self.temp_dir.name) / "auth_profiles" / "test.json"
        assert profile_path.exists()

        LLMRegistry.delete_auth_profile("test")
        assert not profile_path.exists()

    def test_delete_auth_profile_not_found(self):
        """Test deleting non-existent profile raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            LLMRegistry.delete_auth_profile("missing")

    def test_list_auth_profiles_empty(self):
        """Test listing when no profiles exist."""
        assert LLMRegistry.list_auth_profiles() == []

    def test_list_auth_profiles_multiple(self):
        """Test listing multiple profiles."""
        LLMRegistry.save_auth_profile(
            LLMAuth(
                name="p1",
                credentials=self.sample_auth.credentials,
            ),
            override_existing=True,
        )
        LLMRegistry.save_auth_profile(
            LLMAuth(
                name="p2",
                credentials=self.sample_auth.credentials,
            ),
            override_existing=True,
        )
        LLMRegistry.save_auth_profile(
            LLMAuth(
                name="p3",
                credentials=self.sample_auth.credentials,
            ),
            override_existing=True,
        )

        profiles = LLMRegistry.list_auth_profiles()
        assert len(profiles) == 3
        assert set(profiles) == {"p1", "p2", "p3"}

    def test_list_auth_profiles_returns_names_only(self):
        """Test that list_auth_profiles returns just names, not paths."""
        LLMRegistry.save_auth_profile(self.sample_auth)
        profiles = LLMRegistry.list_auth_profiles()
        assert profiles == ["test"]
        assert not any("/" in p or ".json" in p for p in profiles)

    def test_list_auth_profiles_filtered_by_status(self):
        """Test filtering auth profiles by status."""
        missing_auth = LLMAuth(name="missing", credentials={})
        unreadable_auth = LLMAuth(
            name="unreadable",
            credentials=self.sample_auth.credentials,
        )
        LLMRegistry.save_auth_profile(missing_auth, override_existing=True)
        LLMRegistry.save_auth_profile(unreadable_auth, override_existing=True)

        missing_profiles = LLMRegistry.list_auth_profiles(LLMAuthStatus.MISSING)
        unreadable_profiles = LLMRegistry.list_auth_profiles(LLMAuthStatus.UNREADABLE)

        assert missing_profiles == ["missing"]
        assert unreadable_profiles == ["unreadable"]


class TestLLMAuthProfileEndToEnd:
    """End-to-end test for auth profiles applied to LLM instances."""

    @pytest.fixture(autouse=True)
    def setup_auth_profile_e2e(self):
        temp_dir = tempfile.TemporaryDirectory()
        original_method = LLMRegistry._get_auth_profiles_dir
        original_key = os.environ.get("OPENHANDS_ENCRYPTION_KEY")

        def mock_get_auth_profiles_dir():
            path = Path(temp_dir.name) / "auth_profiles"
            path.mkdir(parents=True, exist_ok=True)
            return path

        LLMRegistry._get_auth_profiles_dir = staticmethod(mock_get_auth_profiles_dir)
        os.environ["OPENHANDS_ENCRYPTION_KEY"] = "test-key"

        yield

        LLMRegistry._get_auth_profiles_dir = original_method
        temp_dir.cleanup()
        if original_key is None:
            os.environ.pop("OPENHANDS_ENCRYPTION_KEY", None)
        else:
            os.environ["OPENHANDS_ENCRYPTION_KEY"] = original_key

    @pytest.mark.parametrize("method_name", ["completion", "responses"])
    def test_llm_loads_auth_profile_credentials(self, method_name):
        auth = LLMAuth(name="e2e", credentials={"api_key": SecretStr("sk-e2e")})
        LLMRegistry.save_auth_profile(auth, override_existing=True)

        llm = LLM(model="openai/gpt-4o", usage_id="e2e", auth_profile="e2e")
        messages = [Message(role="user", content=[TextContent(text="Hi")])]

        if method_name == "completion":
            with patch("openhands.sdk.llm.llm.litellm_completion") as mock_completion:
                mock_completion.return_value = create_mock_litellm_response("ok")
                llm.completion(messages=messages)
        else:
            msg = ResponseOutputMessage.model_construct(
                id="m1",
                type="message",
                role="assistant",
                status="completed",
                content=[
                    ResponseOutputText(
                        type="output_text",
                        text="ok",
                        annotations=[],
                    )
                ],
            )
            usage = ResponseAPIUsage(input_tokens=0, output_tokens=0, total_tokens=0)
            resp = ResponsesAPIResponse(
                id="resp123",
                created_at=0,
                output=[msg],
                usage=usage,
                parallel_tool_calls=False,
                tool_choice="auto",
                top_p=None,
                tools=[],
                instructions="",
                status="completed",
            )
            with patch("openhands.sdk.llm.llm.litellm_responses") as mock_responses:
                mock_responses.return_value = resp
                llm.responses(messages=messages)

        assert llm.api_key is not None
        assert isinstance(llm.api_key, SecretStr)
        assert llm.api_key.get_secret_value() == "sk-e2e"
