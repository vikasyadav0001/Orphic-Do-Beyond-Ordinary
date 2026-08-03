"""
=========================
Builds and exposes the n8n Deep Agent.

Responsibilities:
  - Instantiate create_deep_agent with n8n MCP tools + create_n8n_credential
  - Expose get_workflow_agent() for lazy init during initialize_graph()
  - Nothing else — no graph logic, no state, no routing

The Deep Agent is invoked from tools/workflow_tool.py, not from graph.py directly.
interrupt_on={"publish_workflow": True} fires HITL before any workflow goes live.
GraphInterrupt from publish_workflow propagates up through workflow_tool → tool_node
→ compiled_graph exactly like any other interrupt.
"""

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from deepagents.graph import create_deep_agent
from langgraph.graph.state import CompiledStateGraph

from config import get_settings
from memory.graph_checkpointer import pool
from prompts.workfow_agent_prompt import workflow_agent_prompt
from tools.workflow_agent_tools import create_n8n_credential
from utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)

# Module-level — set by get_workflow_agent(), reused after
_workflow_agent: CompiledStateGraph | None = None


async def get_workflow_agent() -> CompiledStateGraph | None:
    """
    Lazy initializer for the n8n Deep Agent.
    Called once from initialize_graph() in graph.py.
    Returns None if n8n is unavailable — callers handle gracefully.

    Why same pool as main graph?
      AsyncPostgresSaver(pool) shares the Postgres connection pool.
      This means the Deep Agent checkpoints to the same DB, and when
      publish_workflow fires interrupt_on, the GraphInterrupt surfaces
      on the same thread_id as the main graph. Resume endpoint works
      identically for both workflow and regular tool interrupts.
    """
    global _workflow_agent

    if _workflow_agent is not None:
        return _workflow_agent

    # Load n8n MCP tools
    n8n_tools = []
    try:
        from protocols.mcp.native_mcp_tool_config import get_workflow_agent_tools
        n8n_tools = await get_workflow_agent_tools()
        logger.info(f"Loaded {len(n8n_tools)} n8n MCP tools for workflow agent")
    except Exception as e:
        logger.warning(f"n8n MCP tools unavailable: {e}")

    if not n8n_tools:
        logger.warning(
            "No n8n MCP tools loaded — workflow agent will have no n8n capabilities. "
            "Ensure n8n is running and N8N_TOKEN is set."
        )

    # create_n8n_credential is a custom tool that handles provider auth
    # and injects tokens into n8n via REST API before workflow creation
    workflow_tools = n8n_tools + [create_n8n_credential]

    try:
        _workflow_agent = create_deep_agent(
            model=ChatOpenAI(
                model="gpt-5-nano",
                api_key=settings.openai_api_key,
                temperature=0,
                streaming=True,
            ),
            tools=workflow_tools,
            system_prompt=workflow_agent_prompt,
            interrupt_on={"publish_workflow": True},
            # checkpointer=AsyncPostgresSaver(pool),
            name="orphic-n8n-workflow-agent",
            
        )
        logger.info("Workflow Deep Agent initialized successfully")
        return _workflow_agent

    except Exception as e:
        logger.error(f"Failed to create workflow Deep Agent: {e}", exc_info=True)
        return None