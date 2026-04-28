from dotenv import load_dotenv
from uuid import uuid4
from typing import List
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.store.base import BaseStore
import memory.long_term_memory as ltm
from memory.long_term_memory import retrieve_memory, store_memory, setup_memory_store
from utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

#Extractor LLM
try:
    extractor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
except Exception as e:
    logger.error(f"Failed to initialize extractor LLM: {e}")
    raise

#class for ignoring the dedupulication of the memories not to store in the store
class MemoryItem(BaseModel):
    text: str = Field(description="Atomic user memory as a short sentence")
    is_new: bool = Field(description="True if this is NEW and should be stored. False if duplicate/already known.")

#extractor state
class MemoryDecesion(BaseModel):
    should_write: bool = Field(description="Whether to store any memories")
    memories: List[MemoryItem] = Field(default_factory=list, description="Atomic user memories to store")


#initialise the extractor
try:
    memory_extractor = extractor_llm.with_structured_output(MemoryDecesion)
except Exception as e:
    logger.error(f"Failed to configure structured output: {e}")
    raise

MEMORY_PROMPT = """You are responsible for maintaining a high-quality long-term memory store for a user.

Your job is NOT to store everything. Your job is to store ONLY information that is:
- Stable over time (won’t change frequently)
- Useful for future interactions
- Explicitly stated by the user (no guessing)

--------------------------------
CURRENT USER DETAILS:
{user_details_content}
--------------------------------

INSTRUCTIONS:

1. Review the user's latest message carefully.

2. Extract ONLY memory-worthy facts from these categories:
   - Identity (name, role, background)
   - Long-term goals (career, financial, learning goals)
   - Ongoing projects (clearly active work)
   - Strong preferences (communication style, tools, habits)
   - Constraints (time, resources, limitations)

3. DO NOT store:
   - Temporary intent ("I am trying this today")
   - One-time actions
   - Questions
   - Emotional states unless persistent
   - Generic or obvious statements
   - Information already captured with same meaning

4. De-duplication rules:
   - If a memory with SAME meaning exists → mark is_new = false
   - If it ADDS meaningful new detail → update existing (is_new = true)
   - Avoid paraphrase duplicates

5. Compression rules:
   - Each memory must be ONE short atomic sentence
   - No filler words
   - No explanation
   - No redundancy

   Bad: "User is someone who really likes working with machine learning models a lot"
   Good: "User works with machine learning models"

6. Truth constraint:
   - Only store facts explicitly stated
   - No inference, no assumptions

7. Output rules:
   - Return a list of structured memory objects
   - If nothing qualifies → return an empty list []

--------------------------------
QUALITY BAR (STRICT):

Before storing, ask:
- Will this still matter after 30 days?
- Will this improve future responses?
- Is this specific and non-obvious?

If NOT → discard.

--------------------------------
"""


#graph start
async def chat_create_memory_node(state: MessagesState, config: RunnableConfig, store : BaseStore):
    try:
        user_id = config["configurable"]["user_id"]
    except KeyError:
        logger.error("user_id not found in config")
        return {}

    namespace = ("memories", user_id)

    try:
        #load existing memories
        existing_items = await retrieve_memory(user_id, query="user preferences and background")
        existing_texts = existing_items  # already List[str]
        user_details_content = "\n".join(f"- {t}" for t in existing_texts) if existing_texts else "(empty)"

        #take the user's last message
        if not state["messages"]:
            logger.warning("No messages in state")
            return {}

        last_msg = state["messages"][-1].content

        try:
            decesion : MemoryDecesion = await memory_extractor.ainvoke(
                [
                    SystemMessage(
                        content=MEMORY_PROMPT.format(user_details_content=user_details_content)
                    ),
                    {"role" : "user", "content" : last_msg}
                ]
            )
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return {}

        if decesion.should_write:
            for mem in decesion.memories:
                if mem.is_new:
                    try:
                        await ltm.store.aput(namespace, str(uuid4()), {"content": mem.text})
                        logger.info(f"Stored new memory: {mem.text}")
                    except Exception as e:
                        logger.error(f"Failed to store memory: {e}")

        return {}
    except Exception as e:
        logger.error(f"Unexpected error in chat_create_memory_node: {e}")
        return {}


builder = StateGraph(MessagesState)
builder.add_node("remember", chat_create_memory_node)
builder.add_edge(START, "remember")
builder.add_edge("remember", END)

graph = builder.compile(store=ltm.store)


