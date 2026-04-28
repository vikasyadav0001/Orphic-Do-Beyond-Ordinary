"""
test_memory_middleware.py
=========================
Tests the MemoryMiddleware in isolation — NO real DB, NO real LLM, NO agent needed.
We mock retrieve_memory so the test runs instantly without any network calls.

Tests:
  1. abefore_agent  → must return {"memory_prompt": <formatted string>}
  2. awrap_model_call → must call handler with the correct SystemMessage injected
"""

import asyncio
import sys
from unittest.mock import AsyncMock, patch, MagicMock
from langchain_core.messages import SystemMessage
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage

# ── Fake memories that would normally come from Postgres vector store ─────────
FAKE_MEMORIES = [
    "User's name is Vikas",
    "User works on Project Orphic",
    "User prefers Python",
]


async def test_abefore_agent():
    """
    Test 1: abefore_agent
    ─────────────────────
    We patch retrieve_memory to return FAKE_MEMORIES so no DB connection is needed.
    Then we call abefore_agent and check that:
      - It returns a dict
      - The dict has "memory_prompt" key
      - The memory_prompt contains the user's name from the fake memories
    """
    print("\n── Test 1: abefore_agent ──────────────────────────────────────")

    # patch the retrieve_memory function inside the middleware module
    # AsyncMock makes it behave like an async function returning FAKE_MEMORIES
    with patch("middleware.memory_middleware.retrieve_memory", new=AsyncMock(return_value=FAKE_MEMORIES)):

        from middleware.memory_middleware import MemoryMiddleware

        middleware = MemoryMiddleware(user_id="test_user")

        # fake state — just needs "messages" key as per AgentState schema
        fake_state = {"messages": []}
        # runtime is not used in abefore_agent so None is fine
        fake_runtime = None

        result = await middleware.abefore_agent(fake_state, fake_runtime)

        # ── Assertions ─────────────────────────────────────────────────
        assert isinstance(result, dict), "❌ abefore_agent must return a dict"
        assert "memory_prompt" in result, "❌ dict must have 'memory_prompt' key"
        assert "Vikas" in result["memory_prompt"], "❌ memory_prompt must contain user's name"
        assert "Project Orphic" in result["memory_prompt"], "❌ memory_prompt must contain project"

        print("✅ abefore_agent returned correct dict with 'memory_prompt'")
        print(f"   Preview: {result['memory_prompt'][:]}...")


async def test_awrap_model_call():
    """
    Test 2: awrap_model_call
    ────────────────────────
    We create a fake ModelRequest that already has "memory_prompt" in its state
    (simulating what abefore_agent would have stored).
    Then we call awrap_model_call and verify:
      - The handler was called
      - The handler received a request with the correct system_message injected
    """
    print("\n── Test 2: awrap_model_call ───────────────────────────────────")

    with patch("middleware.memory_middleware.retrieve_memory", new=AsyncMock(return_value=FAKE_MEMORIES)):

        from middleware.memory_middleware import MemoryMiddleware
        from prompts.system_persona_prompt import get_prompt

        middleware = MemoryMiddleware(user_id="test_user")

        # Build the expected prompt (same way abefore_agent would)
        expected_prompt = get_prompt(FAKE_MEMORIES)

        # fake ModelRequest with state already containing memory_prompt
        fake_request = ModelRequest(
            model=MagicMock(),          # we don't actually call the model
            messages=[],                # empty conversation
            system_message=None,        # starts with no system message
            state={
                "messages": [],
                "memory_prompt": expected_prompt  # ← what abefore_agent would have stored
            },
            runtime=None,
        )

        # captured_request stores what the handler received so we can inspect it
        captured_request = {}

        async def fake_handler(request):
            """Fake LLM handler — records the request it received, returns a dummy response."""
            captured_request["value"] = request
            return ModelResponse(result=[AIMessage(content="I am ORPHIC, your AI assistant")])

        # call awrap_model_call — it should inject the system message then call fake_handler
        await middleware.awrap_model_call(fake_request, fake_handler)

        # ── Assertions ─────────────────────────────────────────────────
        assert "value" in captured_request, "❌ handler was never called"
        injected = captured_request["value"].system_message
        assert injected is not None, "❌ system_message was not injected"
        assert isinstance(injected, SystemMessage), "❌ system_message must be a SystemMessage"
        assert "Vikas" in injected.content, "❌ injected prompt must contain user's name"

        print("✅ awrap_model_call correctly injected SystemMessage into request")
        print(f"   system_message preview: {injected.content[:]}...")


async def main():
    print("=" * 60)
    print("  MemoryMiddleware Isolated Tests")
    print("=" * 60)
    await test_abefore_agent()
    await test_awrap_model_call()
    print("\n" + "=" * 60)
    print("  All tests passed ✅")
    print("=" * 60)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
