# tests/
# Unit and integration tests for the entire system.
# Run with: pytest tests/ -v
# Responsibilities:
#   - test_agents.py        → Tests for individual agent nodes and their outputs
#   - test_tools.py         → Tests for each tool (mocked APIs where needed)
#   - test_orchestrators.py → End-to-end graph execution tests with mock LLMs
#   - test_memory.py        → Tests for checkpointer save/load and vector retrieval
#   - test_api.py           → FastAPI endpoint tests (auth, streaming, thread management)
#   - conftest.py           → Shared pytest fixtures (mock DB, mock LLM, test client)
