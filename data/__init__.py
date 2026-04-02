# data/
# Persistent data storage for the system — NOT application code.
# Responsibilities:
#   - embeddings/   → Pre-computed vector embeddings for semantic search (RAG)
#   - raw/          → Raw source documents (PDFs, TXTs, HTMLs) before processing
#   - transcripts/  → Conversation transcripts, agent trace logs, evaluation results
# This folder is excluded from version control (.gitignore) except for sample test data.
# Populating this folder is done via scripts/ingest_docs.py
