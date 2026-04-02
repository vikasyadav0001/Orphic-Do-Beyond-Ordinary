# Project Orphic — Does Beyond Ordinary

> An autonomous, multimodal, multi-agent AI system built for deep research, multi-step reasoning, and real-world task execution.

---

## Vision

Orphic is a production-grade agentic AI platform. It goes beyond ordinary chatbots by orchestrating multiple specialized AI agents that can plan, reason, use tools, and collaborate — autonomously completing complex, multi-step tasks.

---

## Architecture

```
agents/         All agent types — Planning, Execution, Monitoring
orchestrators/  LangGraph graphs that wire agents, tools, and memory together
cognition/      Reasoning strategies — CoT, ReAct, Reflection, Tree-of-Thought
memory/         Short-term (checkpointer) and long-term (vector) memory
tools/          Web search, code execution, file reading, database queries
modalities/     Text, Vision, and Audio input/output handling
protocols/      MCP (Model Context Protocol) and A2A agent communication
services/       LLM providers, vector stores, embedding APIs
api/            FastAPI — HTTP/WebSocket endpoints for the frontend
db/             PostgreSQL — connection pool, ORM models, migrations
configs/        YAML configs for models, tools, agents
workflows/      Sequential, parallel, and hybrid agent workflows
utils/          Logging, retries, token counting, observability
data/           Embeddings, raw documents, transcripts
tests/          Unit and integration tests
scripts/        Run, evaluate, ingest, benchmark, visualize
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Orchestration | LangGraph |
| LLM Providers | OpenAI GPT, Google Gemini, Anthropic Claude |
| Memory | PostgreSQL + pgvector (LangGraph checkpointer) |
| API | FastAPI + WebSocket |
| Protocols | MCP (Model Context Protocol), A2A |
| Frontend | React / Next.js *(Phase 3)* |
| Search | Tavily API |
| Observability | LangSmith + OpenTelemetry |

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/project-orphic.git
cd project-orphic

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your API keys and database URL

# 5. Run the agent
python scripts/run_agent.py
```

---

## Development Phases

- **Phase 1** ✅ — Project structure, LangGraph core, Postgres checkpointer
- **Phase 2** 🔄 — FastAPI backend, auth, WebSocket streaming
- **Phase 3** ⏳ — React frontend, real-time UI
- **Phase 4** ⏳ — Multi-agent orchestration, tools, MCP integration
- **Phase 5** ⏳ — Multimodal (vision + audio), long-term vector memory

---

## License

MIT
