"""Save and reuse LLM configurations with encrypted profiles.

Profiles let you:
- Store LLM configs (model, temperature, API keys) securely
- Switch between different models without editing code
- Share configurations (API keys encrypted via OPENHANDS_ENCRYPTION_KEY)

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

# Create LLM configuration
llm = LLM(
    model="anthropic/claude-sonnet-4-5-20250929",
    api_key=SecretStr(api_key),
    usage_id="agent",
)

# Save encrypted profile
profile_name = "my-claude-profile"
LLMRegistry.save_profile(profile_name, llm, override_existing=True)
print(f"✓ Saved encrypted profile '{profile_name}' to ~/.openhands/llm_profiles/")

# List profiles
print(f"✓ Available profiles: {LLMRegistry.list_profiles()}")

# Load and use profile
loaded_llm = LLMRegistry.load_profile(profile_name)
print(f"✓ Loaded: {loaded_llm.model}")

agent = Agent(llm=loaded_llm)
print("✓ Agent created with loaded profile")

# Cleanup
LLMRegistry.delete_profile(profile_name)
print(f"✓ Deleted profile '{profile_name}'")
