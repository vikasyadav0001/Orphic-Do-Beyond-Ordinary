from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage
from utils.logger import get_logger
from config import get_settings



logger = get_logger(__name__)
settings = get_settings()

# try:
#     summarization_llm = ChatGroq(
#         model="qwen/qwen3.6-27b",
#         api_key=settings.groq_api_key,
#     )
# except Exception as e:
#     logger.error(f"Failed to initialize summarization LLM: {e}")
#     raise

try:
    summarization_llm = ChatMistralAI(
        model="mistral-small-2506",
        api_key=settings.mistral_api_key,
    )
except Exception as e:
    logger.error(f"Failed to initialize summarization LLM: {e}")
    raise


async def maybe_summarize(messages: list, user_id: str) -> list:
    """
    Trigger: estimated token count > 4000 (1 token ≈ 4 chars)
    Keep:    last 10 messages + 1 summary SystemMessage
    Fallback: hard truncate to last 10 on summarization LLM failure
    """
    total_chars = sum(
        len(str(getattr(m, "content", "") or "")) for m in messages
    )
    estimated_tokens = total_chars // 4

    if estimated_tokens <= 10000:
        return messages

    recent = messages[-10:]
    to_summarize = messages[:-10]

    if not to_summarize:
        return messages

    logger.info(
        f"Summarizing {len(to_summarize)} messages for user {user_id} "
        f"(~{estimated_tokens} estimated tokens)"
    )

    summary_content = "\n".join(
        f"{getattr(m, 'type', 'message')}: {getattr(m, 'content', '')}"
        for m in to_summarize
    )

    try:
        summary_response = await summarization_llm.ainvoke([
            HumanMessage(
                content=(
                    "Summarize the following conversation history concisely. "
                    "Preserve: key decisions, facts established, tools called and "
                    "their results, user preferences expressed, unresolved questions.\n\n"
                    f"{summary_content}"
                )
            )
        ])
        summary_msg = SystemMessage(
            content=f"[Conversation Summary — earlier context]: {summary_response.content}"
        )
        return [summary_msg] + recent

    except Exception as e:
        logger.warning(
            f"Summarization LLM failed for user {user_id}: {e}. "
            f"Falling back to hard truncation."
        )
        return messages[-10:]






