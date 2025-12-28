"""Save and reuse LLM configurations and auth profiles.

Profiles let you:
- Store LLM configs (model, temperature, API keys) securely
- Switch between different models without editing code
- Share configurations (API keys encrypted via OPENHANDS_ENCRYPTION_KEY)

Auth profiles let you:
- Store credentials independently of model settings
- Reuse the same credentials across multiple LLM configs

Security:
- API keys are encrypted before saving to disk
- Without encryption key, secrets are redacted (safe fallback)
- Profiles stored in ~/.openhands/llm_profiles/
- Auth profiles stored in ~/.openhands/auth_profiles/
"""

import os

from pydantic import SecretStr

from openhands.sdk import LLM, Agent, LLMRegistry
from openhands.sdk.llm.llm_auth import LLMAuth


# Encryption key required for secure storage
encryption_key = os.getenv("OPENHANDS_ENCRYPTION_KEY")
assert encryption_key is not None, (
    "OPENHANDS_ENCRYPTION_KEY must be set to encrypt API keys.\n"
    "Generate: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
)

api_key = os.getenv("LLM_API_KEY")
assert api_key is not None, "LLM_API_KEY must be set"

# Create LLM configuration
llm = LLM(
    model="anthropic/claude-sonnet-4-5-20250929",
    api_key=SecretStr(api_key),
    usage_id="agent",
)

# Save encrypted LLM profile
llm_profile_name = "my-claude-profile"
LLMRegistry.save_llm_profile(llm_profile_name, llm, override_existing=True)
print(
    f"✓ Saved encrypted LLM profile '{llm_profile_name}' to ~/.openhands/llm_profiles/"
)

# Save encrypted auth profile
auth_profile_name = "my-claude-auth"
auth_profile = LLMAuth(
    name=auth_profile_name,
    credentials={"api_key": SecretStr(api_key)},
)
LLMRegistry.save_auth_profile(auth_profile, override_existing=True)
print(
    f"✓ Saved encrypted auth profile '{auth_profile_name}' "
    "to ~/.openhands/auth_profiles/"
)

# List profiles
print(f"✓ Available LLM profiles: {LLMRegistry.list_llm_profiles()}")
print(f"✓ Available auth profiles: {LLMRegistry.list_auth_profiles()}")

# Load and use LLM profile
loaded_llm = LLMRegistry.load_llm_profile(llm_profile_name)
print(f"✓ Loaded LLM: {loaded_llm.model}")

# Load and use auth profile
loaded_auth = LLMRegistry.load_auth_profile(auth_profile_name)
print(f"✓ Loaded auth profile: {loaded_auth.name}")

llm_with_auth = LLM(
    model=loaded_llm.model,
    usage_id="agent",
    auth_profile=loaded_auth.name,
)

agent = Agent(llm=loaded_llm)
print("✓ Agent created with loaded LLM profile")

agent_with_auth = Agent(llm=llm_with_auth)
print("✓ Agent created with auth profile")

# ============================================================================
# Registry profile workflow - save multiple LLMs together
# ============================================================================
print("\n=== Registry Profile Demo ===")

# Create registry with multiple LLMs using the same profile with different usage IDs
registry = LLMRegistry()
registry.add_llms_from_profiles(
    {
        "agent": llm_profile_name,  # Load profile with usage_id="agent"
        "title-gen": llm_profile_name,  # Load same profile with usage_id="title-gen"
    }
)
print(
    f"✓ Loaded {len(registry.list_usage_ids())} LLM(s) into registry: "
    f"{registry.list_usage_ids()}"
)

# Save entire registry as a profile
registry_profile_name = "my-multi-llm-setup"
registry.save_registry_profile(registry_profile_name, override_existing=True)
print(
    f"✓ Saved registry profile '{registry_profile_name}' "
    "to ~/.openhands/llm_registry_profiles/"
)

# Load registry profile
loaded_registry = LLMRegistry.load_registry_profile(registry_profile_name)
print(
    f"✓ Loaded registry with {len(loaded_registry.list_usage_ids())} LLM(s): "
    f"{loaded_registry.list_usage_ids()}"
)

# Check credential status for all LLMs
status = loaded_registry.get_llms_status()
print(f"✓ LLM credential status: {status}")

# Cleanup
LLMRegistry.delete_llm_profile(llm_profile_name)
print(f"✓ Deleted LLM profile '{llm_profile_name}'")
LLMRegistry.delete_auth_profile(auth_profile_name)
print(f"✓ Deleted auth profile '{auth_profile_name}'")
LLMRegistry.delete_registry_profile(registry_profile_name)
print(f"✓ Deleted registry profile '{registry_profile_name}'")
