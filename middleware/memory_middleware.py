from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import AgentState, Runtime
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_config
from typing_extensions import NotRequired
from memory.long_term_memory import retrieve_memory
from prompts.system_persona_prompt import get_prompt
from schemas.context import UserContext
from utils.logger import get_logger

logger = get_logger(__name__)

#define the agent state
class MemoryAgentState(AgentState):
    memory_prompt : NotRequired[str]

#configure the agent middleware and provide the agent state
class MemoryMiddleware(AgentMiddleware):
    state_schema = MemoryAgentState

    # Runs ONCE per user turn (before the ReAct loop starts).
    # Fetches the latest memories from the DB and stores in state
    # so awrap_model_call can read them without hitting the DB again.
    async def abefore_agent(self, state, runtime: Runtime[UserContext]):
        user_id = self._resolve_user_id(runtime)
        try:
            memories = await retrieve_memory(user_id=user_id, query="user details")
            prompt = get_prompt(memories)
            logger.debug(f"Loaded {len(memories)} memories for user {user_id}")
            return {"memory_prompt": prompt}
        except Exception as e:
            logger.error(f"Failed to load memories for {user_id}: {e}")
            return {"memory_prompt": get_prompt([])}

    def _resolve_user_id(self, runtime: Runtime[UserContext]) -> str:
        """
        Resolves user_id with a clear priority chain:
          1. runtime.context.user_id  — set when called via API with context=UserContext(...)
          2. config["configurable"]["user_id"] — set in the LangGraph config dict
          3. "default_user" — last-resort fallback, logged as a warning
        """
        # Priority 1: typed context injected at call-site (production API path)
        try:
            if runtime.context is not None and runtime.context.user_id:
                return runtime.context.user_id
        except AttributeError:
            pass

        # Priority 2: LangGraph configurable dict (CLI / test runner path)
        try:
            config: RunnableConfig = get_config()
            user_id = config.get("configurable", {}).get("user_id")
            if user_id:
                logger.debug(f"user_id resolved from configurable: {user_id}")
                return user_id
        except Exception:
            pass

        # Priority 3: fallback — should never reach here in production
        logger.warning(
            "user_id could not be resolved from context or configurable. "
            "Falling back to 'default_user'. Ensure context=UserContext(user_id=...) "
            "is passed to bot.astream() on the API path."
        )
        return "default_user"

    # Runs on every model call within the ReAct loop
    # Reads memory_prompt from state (no DB hit) and injects into system message
    async def awrap_model_call(self, request, handler):
        try:
            prompt = request.state.get("memory_prompt", "")
            return await handler(
                request.override(system_message=SystemMessage(content=prompt))
            )
        except Exception as e:
            logger.error(f"Error in wrap_model_call: {e}")
            return await handler(request)


    # Wraps every tool call — catches MCP errors that handle_tool_error can't
    # Returns a friendly error message to the agent instead of crashing
    async def awrap_tool_call(self, request, handler):
        try:
            return await handler(request)
        except Exception as e:
            from langchain_core.messages import ToolMessage
            return ToolMessage(
                content=f"Tool error: {str(e)}. Try a different tool or approach.",
                tool_call_id=request.tool_call["id"],
                status="error"
            )