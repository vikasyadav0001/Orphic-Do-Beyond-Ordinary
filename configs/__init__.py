# configs/
# YAML-based configuration files for models, tools, agents, and protocols.
# Keeps configuration out of code — change behavior without touching Python files.
# Responsibilities:
#   - models.yaml    → Which LLM to use per agent (model name, temperature, max tokens)
#   - tools.yaml     → Which tools are enabled, their API keys and rate limits
#   - agents.yaml    → Agent definitions (name, role, tools, memory type, prompt path)
#   - protocols.yaml → MCP server/client endpoints and A2A agent registry
# Loaded at startup via a config loader using Pydantic or PyYAML.
