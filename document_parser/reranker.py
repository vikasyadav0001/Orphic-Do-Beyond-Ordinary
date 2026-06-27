"""
Reranking module for the document retrieval pipeline.

Pipeline:
  1. PGVector cosine similarity → wide candidate pool (top-N)
  2. CrossEncoder re-scores all candidates jointly with the query
  3. Return top-k after re-ranking

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 22MB, CPU-friendly, industry-standard for passage reranking
  - Loaded once at module level and reused across all requests (lru_cache)
"""

import asyncio
from functools import lru_cache
from typing import List

from sentence_transformers import CrossEncoder
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Model name constant — easy to swap for a larger model later ──────────────
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# For higher quality at the cost of speed, alternatives:
#   "cross-encoder/ms-marco-MiniLM-L-12-v2"   (larger, better)
#   "BAAI/bge-reranker-base"                    (multilingual)


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """
    Load the CrossEncoder model once and cache it for the process lifetime.
    First call downloads the model (~22MB) from HuggingFace if not cached locally.
    """
    logger.info(f"Loading reranker model: {RERANKER_MODEL}")
    model = CrossEncoder(RERANKER_MODEL)
    logger.info("Reranker model loaded and ready.")
    return model


async def rerank(
    query: str,
    chunks: List[str],
    top_k: int = 5,
) -> List[str]:
    """
    Re-rank a list of text chunks against a query using a CrossEncoder.

    The CrossEncoder reads the query and each chunk TOGETHER (unlike embeddings
    which compare them separately), producing a much more accurate relevance score.

    Args:
        query:  The user's search query.
        chunks: Candidate chunks from the initial vector similarity search.
        top_k:  Number of top-ranked chunks to return after re-ranking.

    Returns:
        The top_k most relevant chunks, ordered by descending relevance score.
    """
    if not chunks:
        return []

    if len(chunks) <= top_k:
        # Not enough candidates to bother re-ranking — return as-is
        return chunks

    try:
        model = get_reranker()

        # Build (query, chunk) pairs — one per candidate
        pairs = [(query, chunk) for chunk in chunks]

        # CrossEncoder.predict is CPU-bound and synchronous.
        # Run in a thread pool so we don't block the async event loop.
        scores: List[float] = await asyncio.to_thread(model.predict, pairs)

        # Zip chunks with their scores, sort descending, slice top_k
        scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        top_chunks = [chunk for _, chunk in scored[:top_k]]

        logger.info(
            f"Reranked {len(chunks)} candidates → kept top {len(top_chunks)}. "
            f"Best score: {scored[0][0]:.4f}, worst kept: {scored[top_k-1][0]:.4f}"
        )

        return top_chunks

    except Exception as e:
        logger.error(f"Reranking failed: {e}. Falling back to original order.", exc_info=True)
        # Graceful fallback — return original chunks unmodified so search still works
        return chunks[:top_k]
