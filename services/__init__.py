# services/
# External service integrations — connections to third-party APIs and platforms.
# Acts as an abstraction layer so agents don't depend directly on external SDK details.
# Responsibilities:
#   - llm.py           → LLM provider clients (OpenAI, Google Gemini, Anthropic Claude)
#   - vector_store.py  → Vector database clients (Pinecone, Weaviate, pgvector)
#   - embeddings.py    → Text embedding generation (OpenAI embeddings, sentence-transformers)
#   - search_api.py    → Tavily / SerpAPI / Brave search service wrappers
#   - storage.py       → Cloud storage (S3, GCS) for files and documents
# Swapping a provider (e.g., OpenAI → Gemini) only requires changing this layer.
