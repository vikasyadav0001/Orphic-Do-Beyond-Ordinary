# Orphic — AI Agent That Does Beyond Ordinary

A production-grade AI agent platform that talks to you, remembers you across sessions, and takes actions on your behalf — creating GitHub repos, managing Notion pages, building n8n workflows, searching the web, and more.

---

## What It Does

- **Talks naturally** — Streams responses word by word via SSE.
- **Remembers you** — Learns your name, profession, preferences. Persists across sessions using vector search.
- **Uses tools** — Connected to GitHub, Notion, Slack, Gmail, Drive, Calendar, Sheets, Docs, and Atlassian (Jira + Confluence) through MCP.
- **Builds workflows** — A dedicated Deep Agent designs, validates, and publishes n8n workflows on your behalf.
- **Human-in-the-loop** — Pauses before publishing workflows or when a provider needs authorization. Resumes when you approve.
- **Handles errors gracefully** — Auth failures, rate limits, and tool errors are caught and surfaced cleanly — never crashes.
- **Manages long conversations** — Automatically summarizes older messages so the conversation never hits token limits.

---

## How It Works

You send a message
↓
Layer 2 rate limit check (chat: 10 req/hr, 30 req/day, 100k tokens/hr)
↓
agent_node loads your conversation history (from Postgres checkpointer)
↓
agent_node fetches your memories (Redis cache → pgvector fallback)
↓
agent_node fetches your connected providers (Redis → DB fallback)
↓
Summarization runs if conversation exceeds ~4000 tokens
↓
LLM decides what to do:
→ If it needs data: calls a tool
→ If it has enough info: generates a response
↓
tool_node Phase 1: validates ALL tool calls (auth + rate limit) before executing any
↓
tool_node Phase 2: executes all tool calls in parallel
↓
If auth is missing or rate limited → interrupt() fires → SSE sends interrupt event
↓
Response streams back token by token
↓
Memory extractor checks if you said anything worth remembering
↓
Conversation saved to Postgres for next time


---

## Architecture

### StateGraph (LangGraph)

The agent is built as a LangGraph `StateGraph` — not `create_react_agent`. This gives full control over every node, edge, and interrupt.

START → agent_node → tool_node → agent_node → ... → END
↑______________|


**`AgentState`**: `messages` (add_messages reducer), `user_id`, `thread_id`

**`agent_node`** responsibilities:
- Fetches memories from Redis (TTL 120s) → pgvector fallback
- Fetches connected providers from Redis (TTL 300s) → DB fallback
- Runs summarization inline via `maybe_summarize()` if needed
- Builds system prompt via `get_prompt(user_memory, connected_providers, available_providers)`
- Calls LLM with full message history

**`tool_node`** responsibilities (two-phase):
- Phase 1: validates ALL remote tool calls — auth check (`resolve_token`) + rate limit check (`check_and_increment`) — interrupts on failure; zero side effects until all checks pass
- Phase 2: executes all tool calls — remote MCP via `execute_mcp_tool()`, native/custom via `ainvoke()`
- Critical: `except GraphInterrupt: raise` before any generic `except Exception` so `interrupt()` from nested tools is never swallowed

### Multi-Agent (Workflow Deep Agent) Pending- Work in progress

Workflow creation uses an Agent-as-Tool pattern:

Main Agent
↓ calls create_workflow tool
↓ sequential auth check per provider
↓ invokes Deep Agent (LangGraph prebuilt)
↓ plans → builds → validates → fixes → publish_workflow (HITL interrupt)


- Deep Agent runs inside `create_workflow` tool — not as a graph node
- Same `thread_id` and `checkpointer` as main graph — interrupts surface on the same thread
- HITL fires before `publish_workflow` via `interrupt_on={"publish_workflow": True}`
- Resume: `POST /chat/resume/{thread_id}` with `{"action": "continue" | "cancel"}`

---

## Project Structure

orchestrators/ → StateGraph definition — agent_node, tool_node, routing
agents/ → Deep Agent (workflow builder)
middleware/ → (legacy — logic moved inline to graph nodes)
memory/ → Long-term memory (pgvector) and conversation checkpointer (Postgres)
prompts/ → System prompt — personality and rules
protocols/mcp/ → MCP schema extraction and tool execution
services/
connections/ → OAuth engine — per-user token storage, refresh, revocation
cache.py → Redis singleton — tokens, connections, memories
rate_limiter.py → Layer 1 (per-provider) + Layer 2 (chat) rate limiting
tools/ → Custom tools: create_workflow, create_n8n_credential, document tools
utils/
agent_helpers.py → resolve_token, fetch_connections_from_db, cancel_all
summarization.py → maybe_summarize, Groq summarization LLM
scripts/ → run_agent.py, clear_checkpoints.py, extract_schemas.py
mcp_schemas/ → schemas.json (committed — loaded at startup, no network call)


### Key Files

| File | What It Does |
|---|---|
| `orchestrators/graph.py` | StateGraph, agent_node, tool_node, stream_response, resume_response |
| `agents/workflow_agent.py` | Deep Agent singleton for n8n workflow creation |
| `tools/workflow_tool.py` | `create_workflow` tool — auth loop + Deep Agent invocation |
| `tools/n8n_credential_tool.py` | `create_n8n_credential` — POSTs credentials to n8n REST API |
| `services/connections/oauth_engine.py` | OAuth2 + API key storage; Redis CSRF state; auto-refresh |
| `services/cache.py` | Redis singleton; key patterns for tokens, connections, memories |
| `services/rate_limiter.py` | Sliding window rate limiting per user per provider + chat-level |
| `memory/long_term_memory.py` | pgvector store — semantic search over user facts |
| `memory/memory_extractor.py` | Extracts memorable facts from conversations (gpt-4o-mini) |
| `memory/graph_checkpointer.py` | Async Postgres connection pool for LangGraph checkpointer |
| `protocols/mcp/remote_mcp_client_config.py` | `execute_mcp_tool()` — raw MCP SDK, streamable_http |
| `protocols/mcp/native_mcp_tool_config.py` | `get_native_mcp_tools()`, `get_n8n_mcp_tools()` |
| `utils/agent_helpers.py` | `resolve_token` (cache-first, never raises), `cancel_all` |
| `utils/summarization.py` | `maybe_summarize()` — Groq Llama, >4000 tokens trigger, keep last 10 |
| `prompts/system_persona_prompt.py` | `get_prompt(user_memory, connected_providers, available_providers)` |
| `mcp_schemas/schemas.json` | Static tool schemas — committed, loaded at startup |
| `scripts/extract_schemas.py` | Dev script — regenerate schemas.json when providers change |
| `scripts/clear_checkpoints.py` | Clears broken conversation state from Postgres |

---

## Tech Stack

| What | Technology |
|---|---|
| LLM (default) | GPT-5 Nano / GPT-5.4 Nano / GPT-5.6 Luna (user-selectable) |
| Vision | Qwen 3.6 (image understanding in document pipeline) |
| Summarization LLM | Groq llama-4-scout-17b |
| Memory Extractor | gpt-4o-mini with structured output |
| Agent Framework | LangGraph StateGraph + LangChain |
| Database | NeonDB (Postgres + pgvector, Singapore region) |
| Conversation Memory | LangGraph AsyncPostgresSaver checkpointer |
| Long-Term Memory | pgvector with OpenAI Embeddings |
| Cache | Redis 7 (tokens, connections, memories, OAuth CSRF state) |
| External Tools | MCP (Model Context Protocol) — raw SDK, streamable_http |
| Workflow Automation | n8n (self-hosted on VM via Docker) |
| Backend | FastAPI + Gunicorn + UvicornWorker |
| Deployment | Azure VM (B2als_v2) + ACR + GitHub Actions CI/CD |
| Frontend | Azure Static Web Apps |
| Reverse Proxy | Nginx (10 req/s rate limit) |
| Cross-Encoder | cross-encoder/ms-marco-MiniLM-L-6-v2 (baked into Docker image) |

---

## Connected Providers

| Provider | MCP Endpoint | What the Agent Can Do |
|---|---|---|
| **GitHub** | `https://api.githubcopilot.com/mcp/` | List repos, create issues, read files, manage PRs |
| **Notion** | `https://mcp.notion.com/mcp` | Search pages, create pages, update content |
| **Slack** | `https://mcp.slack.com/mcp` | Read channels, send messages, search conversations |
| **Gmail** | `https://gmailmcp.googleapis.com/mcp/v1` | Read, send, search emails |
| **Google Drive** | `https://drivemcp.googleapis.com/mcp/v1` | Search, read, manage files |
| **Google Calendar** | `https://calendarmcp.googleapis.com/mcp/v1` | Read and create events |
| **Google Sheets** | `https://sheetsmcp.googleapis.com/mcp/v1` | Read and write spreadsheet data |
| **Google Docs** | `https://docsmcp.googleapis.com/mcp/v1` | Read and edit documents |
| **Atlassian** | `https://mcp.atlassian.com/v1/mcp` | Jira issues, Confluence pages |
| **n8n** | `http://localhost:5678` (VM-local) | Build and publish automation workflows |

All remote providers use OAuth2 Authorization Code flow. Tokens are stored per-user in Postgres and cached in Redis (TTL 300s) with auto-refresh.

---

## Memory

### Short-Term (Conversation History)
- Saved per thread using `AsyncPostgresSaver` (Postgres)
- Full message history available within a session
- When estimated tokens exceed 4000, older messages are summarized via Groq Llama; last 10 messages always kept intact
- Hard truncation fallback if summarization LLM fails

### Long-Term (Facts About You)
- After every message, `memory_extractor` checks if you mentioned something worth saving (name, profession, preferences, etc.)
- Facts stored in pgvector — retrieved via semantic similarity search
- Loaded into every new conversation, even across sessions
- Cached in Redis per user (TTL 120s) to avoid embedding API call on every turn

---

## Rate Limiting

**Layer 1 — Per-Provider (tool-level)**
- Sliding window per user per provider
- Limits: `per_minute` + `per_hour` (configured per provider)
- Checked in `tool_node` Phase 1 before any tool executes

**Layer 2 — Chat-Level**
- 10 requests/hr, 30 requests/day, 100k tokens/hr, 100k tokens/day
- Checked in `chat_router.py` before graph invocation
- Token usage logged after stream via `log_token_usage()`

---

## Human-in-the-Loop (HITL)

The agent pauses and sends an `interrupt` SSE event in two cases:

1. **Provider not connected** — a tool needs auth you haven't granted
2. **Workflow publish approval** — Deep Agent finishes building a workflow and waits for your go-ahead

**Interrupt payload (auth):**
```json
{
  "type": "interrupt",
  "interrupt_type": "auth_required",
  "provider": "notion",
  "provider_name": "Notion",
  "auth_url": "https://...",
  "message": "Connect your Notion account to continue."
}
```

**Resume:**

POST /chat/resume/{thread_id}
{ "action": "continue" | "cancel" }


Chained interrupts are handled — if a workflow needs credentials for multiple providers, each fires in sequence.

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/vikasyadav0001/Orphic-Do-Beyond-Ordinary.git
cd Orphic-Do-Beyond-Ordinary

python -m venv venv
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate          # Windows

pip install -r requirements.txt
```

### 2. Environment variables

```env
# LLM
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key

# Database
DATABASE_URL=your_neondb_connection_string

# Cache
REDIS_URL=your_redis_url

# Providers (optional — agent works without them, just can't use those tools)
GITHUB_TOKEN=your_github_personal_access_token
NOTION_CLIENT_ID=...
NOTION_CLIENT_SECRET=...
SLACK_CLIENT_ID=...
SLACK_CLIENT_SECRET=...
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
ATLASSIAN_CLIENT_ID=...
ATLASSIAN_CLIENT_SECRET=...

# n8n (VM-local)
N8N_API_KEY=your_n8n_api_key

# Tracing (optional)
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=your_project_name
```

### 3. Run the agent

```bash
python -m scripts.run_agent
```

---

## Troubleshooting

**`tool_calls must be followed by tool messages`**
→ Conversation state is broken. Run: `python -m scripts.clear_checkpoints`

**Interrupt event fires but never resumes**
→ Check that `except GraphInterrupt: raise` exists inside every `try/except` in `tool_node` Phase 2 and inside `create_workflow`. A missing re-raise silently swallows the interrupt.

**Google MCP tools return empty / no tools**
→ `MultiServerMCPClient` silently fails for Google endpoints. Use raw MCP SDK (`streamablehttp_client` + `ClientSession` + `session.call_tool()`) directly in `execute_mcp_tool()`.

**Atlassian tools fail**
→ `cloudId` must be fetched at execution time from `/oauth/token/accessible-resources` and cached in Redis (`atlassian_cloud:{user_id}`, TTL 3600s). The SSE endpoint was deprecated June 30 2026 — use streamable_http only.

**Slack OAuth fails**
→ MCP toggle must be explicitly enabled in your Slack app settings. Dynamic Client Registration is not supported — complete the OAuth code exchange manually to get an `xoxp-` user token.

**Summarization stops working / Groq errors**
→ Groq RPD cap is 1,000 req/day. Monitor usage — hard truncation fallback activates automatically but loses context.

**No response from agent (blank SSE stream)**
→ Check the streaming filter in `graph.py` — filter on `langgraph_node == 'agent_node'`, not `'model'`.

---

## Deployment

- **Stack**: FastAPI + Gunicorn on Azure VM, Nginx reverse proxy
- **Registry**: Azure Container Registry
- **CI/CD**: GitHub Actions — test → build → push → SSH deploy → health check
- **n8n**: self-hosted on same VM via docker-compose
- **Frontend**: Azure Static Web Apps
- **Accessible via**: `https://nice-pebble-0de997500.7.azurestaticapps.net/`
- **API**: `https://orphic-api.centralindia.cloudapp.azure.com/` (SSL via Certbot)

---

## License

MIT