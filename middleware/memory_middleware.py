from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import AgentState
from langchain_core.messages import SystemMessage
from typing_extensions import NotRequired
from memory.long_term_memory import retrieve_memory
from prompts.system_persona_prompt import get_prompt
from utils.logger import get_logger

logger = get_logger(__name__)

#define the agent state
class MemoryAgentState(AgentState):
    memory_prompt : NotRequired[str]

#configure the agent middleware and provide the agent state
class MemoryMiddleware(AgentMiddleware):
    state_schema = MemoryAgentState

    def __init__(self, user_id: str):
        self.user_id = user_id

    # Runs ONCE per user turn (before the ReAct loop starts)
    # Fetches the latest memories from the DB and stores in state
    # so awrap_model_call can read them without hitting the DB again
    async def abefore_agent(self, state, runtime):
        try:
            memories = await retrieve_memory(user_id=self.user_id, query="user details")
            prompt = get_prompt(memories)
            # logger.debug(f"Loaded {len(memories)} memories for user {self.user_id}")
            return {"memory_prompt": prompt}
        except Exception as e:
            logger.error(f"Failed to load memories: {e}")
            return {"memory_prompt": get_prompt([])}

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