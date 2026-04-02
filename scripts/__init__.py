# scripts/
# Standalone executable scripts for running, evaluating, and maintaining the system.
# These are NOT imported as modules — they are run directly from the terminal.
# Responsibilities:
#   - run_agent.py       → Launch the agent system locally for development/testing
#   - evaluate.py        → Run evaluation benchmarks and print scores
#   - ingest_docs.py     → Load documents into the vector store for RAG
#   - seed_db.py         → Populate the database with initial data (users, configs)
#   - benchmark.py       → Stress-test agent performance (latency, token usage)
#   - visualize_graph.py → Generate a visual diagram of the LangGraph agent graph
