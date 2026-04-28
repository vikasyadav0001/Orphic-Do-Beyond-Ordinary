<img width="2816" height="1536" alt="image" src="https://github.com/user-attachments/assets/9560abd9-7559-4e72-93ae-292c45486d63" />

# Orphic — AI Agent That Does Beyond Ordinary

An AI agent that can talk to you, remember things about you, and take actions on your behalf — like creating GitHub repos, managing Notion pages, searching the web, and more.



## What It Does

- **Talks naturally** — Have a normal conversation. It streams responses word by word.
- **Remembers you** — Learns your name, profession, preferences over time. Remembers them across sessions.
- **Uses tools** — Connected to GitHub, Notion, Firecrawl (web scraping), and Exa (search) through MCP.
- **Handles errors gracefully** — If a tool fails, it doesn't crash. It tells you what went wrong and tries a different approach.
- **Manages long conversations** — Automatically summarizes older messages so the conversation doesn't hit token limits.

---

## How It Works

```
You send a message
    ↓
Agent loads your conversation history (from Postgres)
    ↓
Middleware fetches your stored memories (name, profession, etc.)
    ↓
Middleware injects memories into the system prompt
    ↓
LLM (GPT-5 Nano) reads everything and decides what to do:
    → If it needs data: calls a tool (GitHub, Notion, etc.)
    → If it has enough info: generates a response
    ↓
Response streams back to you word by word
    ↓
Memory extractor checks if you said anything worth remembering
    ↓
Conversation saved for next time
```

---

## Project Structure

```
orchestrators/     → Agent setup — connects the LLM, tools, memory, and middleware
middleware/        → Runs before/after each LLM call — injects memory, catches errors
memory/            → Long-term memory (vector store) and conversation history (checkpointer)
prompts/           → System prompt — tells the agent who it is and how to behave
protocols/mcp/     → Connects to external tools (GitHub, Notion, Firecrawl, Exa)
scripts/           → Entry points — run the agent, clear old data, run tests
utils/             → Logging setup
```

### Key Files

| File | What It Does |
|---|---|
| `orchestrators/graph.py` | Creates the agent, handles streaming |
| `middleware/memory_middleware.py` | Injects long-term memory + catches tool errors |
| `memory/long_term_memory.py` | Stores and retrieves user facts using vector search |
| `memory/memory_extractor.py` | Picks out important facts from conversations and saves them |
| `memory/graph_checkpointer.py` | Manages the Postgres connection pool |
| `prompts/system_persona_prompt.py` | The agent's personality and rules |
| `protocols/mcp/remote_mcp_client_config.py` | Connects to all 4 external tool servers |
| `scripts/run_agent.py` | Interactive chat loop — run this to talk to the agent |
| `scripts/clear_checkpoints.py` | Clears old conversation data if something breaks |

---

## Tech Stack

| What | Technology |
|---|---|
| LLM | OpenAI GPT-5 Nano |
| Agent Framework | LangGraph + LangChain |
| Database | NeonDB (Postgres) |
| Conversation Memory | LangGraph Checkpointer (Postgres) |
| Long-Term Memory | Vector store with OpenAI Embeddings |
| External Tools | MCP (Model Context Protocol) |
| Connected Services | GitHub, Notion, Firecrawl, Exa |

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/vikasyadav0001/Orphic-Do-Beyond-Ordinary.git
cd Orphic-Do-Beyond-Ordinary

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

### 2. Set up environment variables

Create a `.env` file in the root folder with these keys:

```env
OPENAI_API_KEY=your_openai_key
DATABASE_URL=your_neondb_connection_string

# External tools (optional — agent works without them, just can't use those tools)
GITHUB_TOKEN=your_github_personal_access_token
NOTION_TOKEN=your_notion_integration_token
FIRECRAWL_API_KEY=your_firecrawl_key
EXA_API_KEY=your_exa_key

# Tracing (optional)
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=your_project_name
```

### 3. Run the agent

```bash
python -m scripts.run_agent
```

You'll see an interactive chat. Type your message and press Enter. Type `exit` to quit.

---

## How Memory Works

The agent has two types of memory:

**Short-term (conversation history)**
- Saved per conversation thread using a Postgres checkpointer
- Everything you said in this session is remembered
- When it gets too long, older messages are summarized automatically

**Long-term (facts about you)**
- After every message, the agent checks if you mentioned something worth remembering
- Things like your name, job, preferences get saved to a vector store
- These facts are loaded into every new conversation — even across different sessions
- Only stores things that matter long-term (not temporary stuff like "search for X")

---

## Middleware

The agent uses two middleware layers that run on every turn:

**Memory Middleware** — `middleware/memory_middleware.py`
- Fetches your stored memories from the database (once per turn, not per LLM call)
- Injects them into the system prompt so the agent knows who you are
- Catches tool errors so the agent doesn't crash when a tool call fails

**Summarization Middleware** — Built-in from LangChain
- Monitors conversation length
- When messages exceed 4000 tokens, summarizes older ones
- Keeps the last 10 messages intact

---

## Connected Tools

| Service | What the agent can do with it |
|---|---|
| **GitHub** | List repos, create issues, read files, manage PRs |
| **Notion** | Search pages, create pages, update content |
| **Firecrawl** | Scrape and read any webpage |
| **Exa** | Web search with AI-powered results |

All tools connect through MCP (Model Context Protocol) — an open standard for connecting AI agents to external services.

---

## Troubleshooting

**Agent crashes with "tool_calls must be followed by tool messages"**
→ Old conversation data is broken. Run: `python -m scripts.clear_checkpoints`

**Notion tools return empty results**
→ Check that your `NOTION_TOKEN` is valid and the integration has access to your workspace pages.

**Agent picks the wrong tool or guesses wrong parameters**
→ The system prompt has rules to prevent this, but it can still happen. The middleware catches errors and lets the agent retry.

**No response from agent (blank output)**
→ Check the streaming filter in `graph.py` — it should filter on `langgraph_node == 'model'`.

---

## License

MIT
