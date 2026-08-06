from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from utils.logger import get_logger
from config import get_settings

logger = get_logger(__name__)
settings = get_settings()

summarizer = None

try:
    if settings.mistral_api_key:
        model_instance = ChatMistralAI(
            model="mistral-small-2506",
            api_key=settings.mistral_api_key,
        )
    else:
        model_instance = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=settings.openai_api_key,
        )

    summarizer = SummarizationMiddleware(
        model=model_instance,
        trigger=("tokens", 10000),
        keep=("messages", 10),
    )
except Exception as e:
    logger.error(f"Failed to initialize SummarizationMiddleware: {e}")


async def maybe_summarize(messages: list, user_id: str) -> list:
    """
    Delegates conversation history summarization to LangChain's built-in SummarizationMiddleware.
    Guarantees tool-call boundary safety and token-based triggering.
    """
    if not messages or len(messages) <= 10 or summarizer is None:
        return messages

    try:
        res = await summarizer.abefore_model({"messages": messages}, runtime=None)
        if res and "messages" in res:
            logger.info(f"SummarizationMiddleware condensed message history for user {user_id}")
            return [
                m for m in res["messages"]
                if getattr(m, "type", "") != "remove" and type(m).__name__ != "RemoveMessage"
            ]
    except Exception as e:
        logger.error(f"Summarization error for user {user_id}: {e}", exc_info=True)

    return messages








