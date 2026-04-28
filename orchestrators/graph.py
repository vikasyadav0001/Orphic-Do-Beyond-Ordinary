from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from memory.graph_checkpointer import pool, setup_db
from protocols.mcp.remote_mcp_client_config import get_mcp_tools
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from dotenv import load_dotenv
from memory.long_term_memory import setup_memory_store, retrieve_memory
from prompts.system_persona_prompt import get_prompt
from middleware.memory_middleware import MemoryMiddleware
from langchain.agents.middleware import SummarizationMiddleware
from utils.logger import get_logger
import os

load_dotenv()
logger = get_logger(__name__)

# Validate OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is required")

# LLM — module level (instantiated once, reused across requests)
try:
    llm = ChatOpenAI(model='gpt-5-nano', streaming=True)
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    raise


# Lazy bot — created once on first request, reused after
_bot = None

async def get_bot():
    """
    Initializes the ReAct agent on first call.
    - Opens the Postgres connection pool
    - Loads all MCP tools (GitHub, Firecrawl, Exa)
    - Compiles the agent with a checkpointer for persistent memory
    """
    global _bot
    if _bot is not None:
        return _bot

    try:
        # print("[1/4] Setting up Postgres...", flush=True)
        await setup_db()                           # open DB pool + create schema
        store = await setup_memory_store()
        # print("[2/4] Connecting to MCP servers...", flush=True)
        checkpointer = AsyncPostgresSaver(pool)

        try:
            mcp_tools = await get_mcp_tools()     # load tools from MCP servers
            logger.info(f"Loaded {len(mcp_tools)} MCP tools")
        except Exception as e:
            logger.warning(f"MCP tools loading failed, continuing without tools: {e}")
            mcp_tools = []

        _bot = create_agent(
            model=llm,
            tools=mcp_tools,
            checkpointer=checkpointer,
            store=store,
            middleware=[
                MemoryMiddleware(user_id="default_user"),
                SummarizationMiddleware(
                    model="gpt-4o-mini",
                    trigger=("tokens", 4000),
                    keep=("messages", 10)
                )
            ]
        )
        logger.info("Agent initialized successfully")
        return _bot
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        raise RuntimeError(f"Agent initialization failed: {e}") from e


async def stream_response(user_message: str, thread_id: str):
    """
    Async generator that streams individual tokens from the ReAct agent.
    Thread ID scopes memory to a specific conversation.
    """
    try:
        bot = await get_bot()
    except Exception as e:
        logger.error(f"Cannot stream: Agent not initialized - {e}")
        yield "Sorry, I'm having trouble starting up. Please try again later."
        return

    config = {
        'configurable': {"thread_id": thread_id},
        'metadata': {"thread_id": thread_id},
        "run_name": "chat_turn"
    }

    try:
        async for event in bot.astream_events(
            {"messages": [HumanMessage(content=user_message)]},
            config=config,
            version='v2'
        ):
            try:
                if (event['event'] == 'on_chat_model_stream' and event.get('metadata', {}).get('langgraph_node') == 'model'):
                    # print(f"DEBUG NODE: {event.get('metadata', {}).get('langgraph_node')}", flush=True)
                    chunk = event['data'].get('chunk')
                    if chunk and hasattr(chunk, 'content') and chunk.content:
                        yield chunk.content
            except Exception as e:
                logger.error(f"Error processing event chunk: {e}")
                continue
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield "\n\n[Sorry, an error occurred. Please try again.]"