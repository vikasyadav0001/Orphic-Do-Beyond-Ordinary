# utils/
# Shared utility functions and helpers used across the entire system.
# These are stateless, reusable helpers with no business logic.
# Responsibilities:
#   - logger.py        → Structured logging setup (JSON logs, log levels, trace IDs)
#   - retry.py         → Exponential backoff retry decorator for API calls
#   - token_counter.py → Count tokens before sending to LLM to avoid context overflow
#   - timer.py         → Measure and log execution time of agent steps
#   - observability.py → LangSmith / OpenTelemetry tracing integration
#   - helpers.py       → Miscellaneous text/data helpers (truncate, slugify, etc.)
