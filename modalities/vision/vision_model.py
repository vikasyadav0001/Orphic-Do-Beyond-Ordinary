import os
import base64
import asyncio
import json
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from functools import lru_cache
from config import get_settings
from langchain_core.messages import HumanMessage, SystemMessage
from document_parser.doc_parser_rag import compute_hash
from services.cache import get_redis
from utils.logger import get_logger

logger = get_logger(__name__)
env = get_settings()



@lru_cache
def get_vision_llm() -> ChatGroq:

    
    # return ChatMistralAI(
    # model="mistral-small-2506",
    # api_key=get_settings().mistral_api_key,
    # )
    return ChatGroq(
        model="qwen/qwen3.6-27b",
        api_key=env.groq_api_key,
        reasoning_format='hidden',
        temperature=0.5,
        max_tokens=4096
    )


async def analyse_image(image_path: str, user_id: str) -> tuple[str, bool]:
    if not os.path.exists(image_path):
        return f"Error: Image path does not exist: {image_path}", False  # ← always tuple

    # ── Dedup check ─────────────────────────────────────────────────
    file_hash = compute_hash(image_path)
    cache_key = f"image_analysis:{user_id}:{file_hash}"

    redis = await get_redis()
    cached_result = await redis.get(cache_key)

    if cached_result:
        logger.info(f"Image already analyzed. Returning cached result.")
        return cached_result, True  

    # ── MIME type ────────────────────────────────────────────────────
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    mime_type = mime_map.get(ext)
    if not mime_type:
        return f"Error: Unsupported file type '{ext}'"

    prompt = """
        Analyze this image with maximum precision. Provide a detailed and structured breakdown of its contents by following these guidelines:

        + Note: The input image may be preprocessed or tiled into panels. Describe the original subject naturally without referring to panel layouts, slices, or vertical collages.
        + Avoid repetitive introductory templates (e.g. DO NOT start responses with "The image is a vertical collage composed of three..."). Start section 1 directly with a unique description of the visual scene.
       
        Do not speculate, make assumptions, or hallucinate. Base your description strictly on what is visually verifiable in the image.

        At last ask the user 2-3 follow up questions, that what exactly they want to get described, and respond the output in sections not as generic text.
    """

    try:
        logger.info(f"Analysing image: {image_path}")

        with open(image_path, "rb") as img:
            image_data = base64.b64encode(img.read()).decode("utf-8")

        image_url = f"data:{mime_type};base64,{image_data}"
        vision_llm = get_vision_llm()
        vision_llm.name
        message = HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": image_url}},
            ])

        response = await vision_llm.ainvoke([SystemMessage(content=prompt)] + [message])
        result = response.content

        # ── Cache and persist ────────────────────────────────────────
        await redis.set(cache_key, result, ex=604800)
        logger.info(f"Image analysis cached for hash {file_hash[:12]}...")

        return result, False

    except Exception as e:
        logger.error(f"Error analyzing image: {e}", exc_info=True)
        return f"Error analyzing image: {str(e)}", False


if __name__ == "__main__":
    image_path = "/home/dell/Desktop/Orphic-Do-Beyond-Ordinary/uploads/Screenshot from 2026-07-13 18-44-04.png"
    result, _ = asyncio.run(analyse_image(image_path=image_path, user_id="default-user1"))
    print(result)