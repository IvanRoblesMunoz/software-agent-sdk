import os

from pydantic import SecretStr

from openhands.sdk import (
    LLM,
    Agent,
    Conversation,
    Event,
    LLMConvertibleEvent,
    LLMRegistry,
    get_logger,
)
from openhands.sdk.tool import Tool
from openhands.tools.terminal import TerminalTool


logger = get_logger(__name__)

# Configure LLM using LLMRegistry
api_key = os.getenv("LLM_API_KEY")
assert api_key is not None, "LLM_API_KEY environment variable is not set."
model = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-5-20250929")
base_url = os.getenv("LLM_BASE_URL")

# Create LLM instance
main_llm = LLM(
    usage_id="agent",
    model=model,
    base_url=base_url,
    api_key=SecretStr(api_key),
)

# Create LLM registry and add the LLM
llm_registry = LLMRegistry()
llm_registry.add(main_llm)

# Get LLM from registry
llm = llm_registry.get("agent")

# Tools
cwd = os.getcwd()
tools = [Tool(name=TerminalTool.name)]

# Agent
agent = Agent(llm=llm, tools=tools)

llm_messages = []  # collect raw LLM messages


def conversation_callback(event: Event):
    if isinstance(event, LLMConvertibleEvent):
        llm_messages.append(event.to_llm_message())


conversation = Conversation(
    agent=agent, callbacks=[conversation_callback], workspace=cwd
)

conversation.send_message("Please echo 'Hello!'")
conversation.run()

print("=" * 100)
print("Conversation finished. Got the following LLM messages:")
for i, message in enumerate(llm_messages):
    print(f"Message {i}: {str(message)[:200]}")

print("=" * 100)
print(f"LLM Registry usage IDs: {llm_registry.list_usage_ids()}")

# ============================================================================
# LLM Profile Persistence
# ============================================================================
print("\n" + "-" * 40)
print("DEMONSTRATING LLM PROFILES")
print("-" * 40)

# IMPORTANT:
# When 'expose_secrets=False', the 'OPENHANDS_ENCRYPTION_KEY' MUST be set.
# Otherwise, the profile would be redacted (masked) and unusable, and
# save_profile() will raise a ValueError to prevent data loss.
if not os.getenv("OPENHANDS_ENCRYPTION_KEY"):
    os.environ["OPENHANDS_ENCRYPTION_KEY"] = "demo-key-for-encrypted-profiles"

profile_name = "example-agent-profile"

# 1. Save profile (Automatically encrypted because key is set)
LLMRegistry.save_profile(
    name=profile_name,
    llm=main_llm,
    expose_secrets=False,  # Enforce encryption/redaction logic
    override_existing=True,  # Explicitly allow overwriting
)
print(f"✓ Saved encrypted profile '{profile_name}' to ~/.openhands/llm_profiles/")

# 2. List all available profiles
print(f"✓ Available profiles: {LLMRegistry.list_profiles()}")

# 3. Load profile back into an LLM (Decryption is automatic if key is set)
loaded_llm = LLMRegistry.load_profile(profile_name)
print(f"✓ Loaded: {loaded_llm.model} (Usage: {loaded_llm.usage_id})")

# 4. Use the loaded configuration for a new agent
new_agent = Agent(llm=loaded_llm)
print("✓ Created new agent from loaded configuration.")

# 5. Cleanup
LLMRegistry.delete_profile(profile_name)
print(f"✓ Deleted profile '{profile_name}'")

# Report cost
cost = llm.metrics.accumulated_cost
print(f"\nEXAMPLE_COST: {cost}")
