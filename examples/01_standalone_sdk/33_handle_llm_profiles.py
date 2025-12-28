"""Save and reuse LLM configurations with encrypted profiles.

Profiles let you:
- Store LLM configs (model, temperature, API keys) securely
- Switch between different models without editing code
- Share configurations (API keys encrypted via OPENHANDS_ENCRYPTION_KEY)
- Load multiple profiles into a registry at once

Security:
- API keys are encrypted before saving to disk
- Without encryption key, secrets are redacted (safe fallback)
- Profiles stored in ~/.openhands/llm_profiles/
"""

import os

from pydantic import SecretStr

from openhands.sdk import LLM, Agent, LLMRegistry


# Encryption key required for secure storage
encryption_key = os.getenv("OPENHANDS_ENCRYPTION_KEY")
assert encryption_key is not None, (
    "OPENHANDS_ENCRYPTION_KEY must be set to encrypt API keys.\n"
    "Generate: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
)

api_key = os.getenv("LLM_API_KEY")
assert api_key is not None, "LLM_API_KEY must be set"

# ============================================================================
# Single LLM profile workflow
# ============================================================================
print("=== Single LLM Profile Demo ===")

# Create LLM configuration
llm = LLM(
    model="anthropic/claude-sonnet-4-5-20250929",
    api_key=SecretStr(api_key),
    usage_id="agent",
)

# Save encrypted profile
LLMRegistry.save_llm_profile("my-claude-profile", llm, override_existing=True)
print("✓ Saved encrypted profile 'my-claude-profile' to ~/.openhands/llm_profiles/")

# List profiles
print(f"✓ Available profiles: {LLMRegistry.list_llm_profiles()}")

# Load profile and optionally override usage_id
loaded_llm = LLMRegistry.load_llm_profile("my-claude-profile", usage_id="custom-agent")
print(f"✓ Loaded: {loaded_llm.model} (usage: {loaded_llm.usage_id})")

agent = Agent(llm=loaded_llm)
print("✓ Agent created with loaded profile")

# Cleanup
LLMRegistry.delete_llm_profile("my-claude-profile")
print("✓ Deleted profile 'my-claude-profile'\n")

# ============================================================================
# Registry bulk loading workflow (using existing example profiles)
# ============================================================================
print("=== Registry Bulk Loading Demo ===")

# Create registry and load multiple profiles at once
registry = LLMRegistry()
registry.add_llms_from_profiles(
    {
        "agent": "claude-sonnet",  # Load claude-sonnet.json with usage_id="agent"
        "title-gen": "gpt-4o",  # Load gpt-4o.json with usage_id="title-gen"
    }
)

print(f"✓ Loaded {len(registry.list_usage_ids())} LLMs into registry")
print(f"✓ Usage IDs: {registry.list_usage_ids()}")

# Access individual LLMs by usage_id
agent_llm = registry.get("agent")
print(f"✓ Agent LLM: {agent_llm.model} (usage: {agent_llm.usage_id})")

# Save entire registry as a profile
registry.save_registry_profile("my-multi-llm-setup", override_existing=True)
print(
    "✓ Saved registry profile 'my-multi-llm-setup' "
    "to ~/.openhands/llm_registry_profiles/"
)

# Load registry profile
loaded_registry = LLMRegistry.load_registry_profile("my-multi-llm-setup")
print(
    f"✓ Loaded registry with {len(loaded_registry.list_usage_ids())} LLMs: "
    f"{loaded_registry.list_usage_ids()}"
)

# Cleanup
LLMRegistry.delete_registry_profile("my-multi-llm-setup")
print("✓ Deleted registry profile 'my-multi-llm-setup'")
