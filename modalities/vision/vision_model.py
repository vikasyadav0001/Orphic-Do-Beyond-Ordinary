import os
import base64
import asyncio
import json
from langchain_groq import ChatGroq
from functools import lru_cache
from config import get_settings
from langchain_core.messages import HumanMessage
from document_parser.doc_parser_rag import compute_hash
from utils.logger import get_logger

logger = get_logger(__name__)
env = get_settings()

# ── Persistent dedup store ──────────────────────────────────────────
IMAGE_CACHE_FILE = "./uploads/.image_cache.json"

# Functions must be defined BEFORE they are called at module level
def _load_cache() -> dict:
    """Load hash → analysis cache from disk."""
    if os.path.exists(IMAGE_CACHE_FILE):
        try:
            with open(IMAGE_CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_cache(cache: dict) -> None:
    """Persist cache to disk."""
    try:
        with open(IMAGE_CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception as e:
        logger.warning(f"Failed to persist image cache: {e}")

# In-memory cache, loaded once at startup (after _load_cache is defined)
_image_analysis_cache: dict = _load_cache()
_cache_lock = asyncio.Lock()



@lru_cache
def get_vision_llm() -> ChatGroq:
    return ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        api_key=env.groq_api_key
    )


async def analyse_image(image_path: str, user_id: str) -> tuple[str, bool]:
    if not os.path.exists(image_path):
        return f"Error: Image path does not exist: {image_path}", False  # ← always tuple

    # ── Dedup check ─────────────────────────────────────────────────
    file_hash = compute_hash(image_path)
    cache_key = f"{user_id}:{file_hash}"

    if cache_key in _image_analysis_cache:
        logger.info(f"Image already analyzed. Returning cached result.")
        return _image_analysis_cache[cache_key], True  # ← read with cache_key, not file_hash

    # ── MIME type ────────────────────────────────────────────────────
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    mime_type = mime_map.get(ext)
    if not mime_type:
        return f"Error: Unsupported file type '{ext}'"

    prompt = """
        Analyze this image with maximum precision. Provide a structured breakdown of its contents by following these guidelines:

        1. Overall Summary: Describe the entire scene or subject in 2-3 sentences. What is the core context?
        2. Key Elements & Subjects: List the main objects, people, or text present. Include details about their placement, colors, and textures.
        3. Text Extraction (OCR): If there is any visible text, signs, labels, or handwriting, transcribe it exactly as it appears. If there is no text, state "None".
        4. Technical/Aesthetic Details: Note the lighting (e.g., natural, harsh, dim), style (e.g., photo, illustration, screenshot), and mood.
        5. Critical Details & Anomalies: Identify any small, subtle, or unusual elements that a casual viewer might miss.

        Do not speculate, make assumptions, or hallucinate. Base your description strictly on what is visually verifiable in the image.
    """

    try:
        logger.info(f"Analysing image: {image_path}")

        with open(image_path, "rb") as img:
            image_data = base64.b64encode(img.read()).decode("utf-8")

        image_url = f"data:{mime_type};base64,{image_data}"
        vision_llm = get_vision_llm()

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        )

        response = await vision_llm.ainvoke([message])
        result = response.content

        # ── Cache and persist ────────────────────────────────────────
        async with _cache_lock:
            _image_analysis_cache[cache_key] = result
            _save_cache(_image_analysis_cache)
            logger.info(f"Image analysis cached for hash {file_hash[:12]}...")

        return result, False

    except Exception as e:
        logger.error(f"Error analyzing image: {e}", exc_info=True)
        return f"Error analyzing image: {str(e)}", False


if __name__ == "__main__":
    image_path = "/home/dell/Desktop/Orphic-Do-Beyond-Ordinary/image.png"
    result, _ = asyncio.run(analyse_image(image_path=image_path))
    print(result)