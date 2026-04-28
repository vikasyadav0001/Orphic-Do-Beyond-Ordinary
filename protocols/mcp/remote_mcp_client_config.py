"""
LangGraph + LangChain MCP Adapters — REMOTE-ONLY Server Configuration
======================================================================
Every server here is a hosted HTTP endpoint or brdge remote servers. Zero local processes.
npx, uvx, subprocesses. Just a URL + token.while for notion bridge remote server ussed via npx


"""
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import os
from utils.logger import get_logger
import asyncio
load_dotenv()

logger = get_logger(__name__)

# Validate required tokens
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
EXA_API_KEY = os.getenv("EXA_API_KEY")
NOTION_TOKEN = os.getenv('NOTION_TOKEN')

missing = [k for k, v in {
    "GITHUB_TOKEN": GITHUB_TOKEN,
    "FIRECRAWL_API_KEY": FIRECRAWL_API_KEY,
    "EXA_API_KEY": EXA_API_KEY,
    "NOTION_TOKEN": NOTION_TOKEN
}.items() if not v]

if missing:
    logger.warning(f"Missing MCP tokens: {', '.join(missing)}. Some tools may not work.")

# ── MCP server connection config ──────────────────────────────────────────────
# Defined here as a plain dict — the actual client is built inside the async
# function below, NOT at module level.
MCP_CONFIG = {}

if GITHUB_TOKEN:
    MCP_CONFIG["github"] = {
        "transport": "http",
        "url": "https://api.githubcopilot.com/mcp/",
        "headers": {"Authorization": f"Bearer {GITHUB_TOKEN}"},
    }

if FIRECRAWL_API_KEY:
    MCP_CONFIG["firecrawl"] = {
        "transport": "http",
        "url": "https://mcp.firecrawl.dev/v2/mcp",
        "headers": {"Authorization": f"Bearer {FIRECRAWL_API_KEY}"},
    }

if EXA_API_KEY:
    MCP_CONFIG["exa"] = {
        "transport": "http",
        "url": "https://mcp.exa.ai/mcp",
        "headers": {"x-api-key": EXA_API_KEY},
    }

if NOTION_TOKEN:
    MCP_CONFIG["notion"] = {
        "transport" : "stdio",
        "command" : "npx",
        "args":["-y", "mcp-remote", "https://mcp.notion.com/mcp"]
    }
 
 
# ── Tool loader ───────────────────────────────────────────────────────────────
async def get_mcp_tools():
    """
    Build the MCP client and return all tools from connected servers.
    FIX 6: client is constructed INSIDE the async function, not at module level.
    """
    if not MCP_CONFIG:
        logger.warning("No MCP servers configured")
        return []

    logger.info(f"Connecting to {len(MCP_CONFIG)} MCP server(s)...")

    connected_tools = []
    failed_servers = []

    # Try each server individually to isolate failures
    for server_name, config in MCP_CONFIG.items():
        try:
            logger.info(f"Connecting to {server_name}...")
            client = MultiServerMCPClient({server_name: config})
            tools = await client.get_tools()

            # for t in tools:
            #     t.handle_tool_error = True

            connected_tools.extend(tools)
            logger.info(f"Connected to {server_name}: {len(tools)} tools loaded")
        except Exception as e:
            logger.error(f"Failed to connect to {server_name}: {e}")
            failed_servers.append(server_name)

    if failed_servers:
        logger.warning(f"Failed to connect to servers: {', '.join(failed_servers)}")

    logger.info(f"Total MCP tools loaded: {len(connected_tools)}")
    return connected_tools


if __name__ == "__main__":
    asyncio.run(get_mcp_tools())



















# ─────────────────────────────────────────────────────────────────────────────
# AUTH SUMMARY — what you need in your .env
#
#  GITHUB_TOKEN✅          → GitHub PAT (scopes: repo, read:org, read:user)
#                          https://github.com/settings/tokens
#
#  SLACK_BOT_TOKEN       → Slack Bot Token (xoxb-...)
#                          Create app at https://api.slack.com/apps
#                          Enable MCP: App Settings → Agents & AI Apps → MCP toggle ON
#                          Add scopes: channels:read, channels:history, chat:write,
#                                      search:read, users:read, canvases:write
#
#  LINEAR_API_KEY        → Linear API Key
#                          https://linear.app/YOUR-TEAM/settings/api
#
#  NOTION_TOKEN✅          → Notion Integration Token (secret_...)
#                          https://www.notion.so/my-integrations → New integration
#                          Then share each page/DB with your integration
#
#  ATLASSIAN_TOKEN       → Atlassian API Token (base64: email:token)
#                          https://id.atlassian.com/manage-profile/security/api-tokens
#                          Encode: base64("your@email.com:your_api_token")
#
#  STRIPE_API_KEY        → Stripe Secret Key (sk_live_... or sk_test_...)
#                          https://dashboard.stripe.com/apikeys
#
#  SENTRY_TOKEN          → Sentry Auth Token
#                          https://sentry.io/settings/account/api/auth-tokens/
#
#  OPENAI_API_KEY        → Your LLM provider key
# ─────────────────────────────────────────────────────────────────────────────



# =============================================================================
# REMOTE MCP SERVERS — all transport: "http" (streamable-http)
# =============================================================================

# REMOTE_MCP_SERVERS = {

#     # ── GitHub ────────────────────────────────────────────────────────────────
#     # Official hosted MCP by GitHub. No local process needed.
#     # Tools: search repos, read/create issues, PRs, commits, code search, etc.
#     # Docs: https://docs.github.com/en/copilot/using-github-copilot/using-github-mcp-server
#     "github": {
#         "transport": "http",
#         "url": "https://api.githubcopilot.com/mcp/",
#         "headers": {
#             "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}",
#         }
#     }

    # # ── Slack ─────────────────────────────────────────────────────────────────
    # # Official hosted MCP by Slack. Requires a registered Slack app.
    # # Steps before using:
    # #   1. Create app at https://api.slack.com/apps
    # #   2. Go to Agents & AI Apps → toggle MCP ON
    # #   3. Add OAuth scopes listed in the env section above
    # #   4. Install app to workspace → copy Bot User OAuth Token (xoxb-...)
    # # Tools: search messages, read channels/threads, send messages, manage canvases
    # # Docs: https://docs.slack.dev/ai/slack-mcp-server
    # "slack": {
    #     "transport": "http",
    #     "url": "https://slack.com/api/mcp",
    #     "headers": {
    #         "Authorization": f"Bearer {os.getenv('SLACK_BOT_TOKEN')}",
    #     },
    # },

    # # ── Linear ────────────────────────────────────────────────────────────────
    # # Official hosted MCP by Linear. Supports both API key (simpler) and OAuth.
    # # Using API key here — get it from: Linear → Settings → Security & Access → API
    # # Tools: find/create/update issues, projects, cycles, comments, team members
    # # Docs: https://linear.app/docs/mcp
    # "linear": {
    #     "transport": "http",
    #     "url": "https://mcp.linear.app/mcp",
    #     "headers": {
    #         "Authorization": f"Bearer {os.getenv('LINEAR_API_KEY')}",
    #     },
    # },

    # # ── Notion ────────────────────────────────────────────────────────────────
    # # Official hosted MCP by Notion. Uses an integration token (not OAuth).
    # # IMPORTANT: You must share each Notion page/DB with your integration manually.
    # #   Go to any Notion page → ··· menu → Connections → Add your integration
    # # Tools: search, read/create/update pages and databases, manage blocks
    # # Docs: https://developers.notion.com/docs/mcp
    # "notion": {
    #     "transport": "http",
    #     "url": "https://mcp.notion.com/mcp",
    #     "headers": {
    #         "Authorization": f"Bearer {os.getenv('NOTION_TOKEN')}",
    #     },
    # },

    # # ── Atlassian — Jira + Confluence ─────────────────────────────────────────
    # # Official hosted MCP by Atlassian. Uses HTTP Basic Auth via base64 encoding.
    # # Encode your credentials first:
    # #   import base64
    # #   token = base64.b64encode(b"your@email.com:your_api_token").decode()
    # # Then set ATLASSIAN_TOKEN=<that base64 string> in your .env
    # # Tools: create/read/update Jira issues, search Confluence pages, manage sprints
    # # Docs: https://www.atlassian.com/blog/announcements/atlassian-mcp
    # # NOTE: Atlassian's server currently speaks SSE — use "sse" transport below
    # "atlassian": {
    #     "transport": "sse",                            # Atlassian uses SSE, not streamable-http yet
    #     "url": "https://mcp.atlassian.com/v1/sse",
    #     "headers": {
    #         "Authorization": f"Basic {os.getenv('ATLASSIAN_TOKEN')}",
    #     },
    # },

    # # ── Stripe ────────────────────────────────────────────────────────────────
    # # Official hosted MCP by Stripe. Use test key (sk_test_...) for dev.
    # # Tools: list/create customers, charges, subscriptions, invoices, products
    # # Docs: https://docs.stripe.com/building-with-llms/mcp
    # "stripe": {
    #     "transport": "http",
    #     "url": "https://mcp.stripe.com/",
    #     "headers": {
    #         "Authorization": f"Bearer {os.getenv('STRIPE_API_KEY')}",
    #     },
    # },

    # # ── Sentry ────────────────────────────────────────────────────────────────
    # # Official hosted MCP by Sentry. Auth token from account settings.
    # # Tools: list issues/errors, get event details, manage projects and teams
    # # Docs: https://docs.sentry.io/product/sentry-basics/integrate-frontend/mcp/
    # "sentry": {
    #     "transport": "http",
    #     "url": "https://mcp.sentry.io/",
    #     "headers": {
    #         "Authorization": f"Bearer {os.getenv('SENTRY_TOKEN')}",
    #     },
    # },

# }







# =============================================================================
# TRANSPORT CHEAT SHEET
# =============================================================================
#
#  Server      URL                                   Transport   Auth
#  ──────────────────────────────────────────────────────────────────────────
#  GitHub      https://api.githubcopilot.com/mcp/    http        Bearer PAT
#  Slack       https://slack.com/api/mcp             http        Bearer xoxb-...
#  Linear      https://mcp.linear.app/mcp            http        Bearer API key
#  Notion      https://mcp.notion.com/mcp            http        Bearer secret_...
#  Atlassian   https://mcp.atlassian.com/v1/sse      sse         Basic base64(email:token)
#  Stripe      https://mcp.stripe.com/               http        Bearer sk_...
#  Sentry      https://mcp.sentry.io/                http        Bearer auth_token
#
# =============================================================================
# COMMON ERRORS
# =============================================================================
#
#  ❌ 401 Unauthorized
#     → Token is wrong, expired, or missing required scopes.
#       Double-check the env var name matches exactly.
#
#  ❌ 403 Forbidden (Slack)
#     → The Slack app doesn't have the required OAuth scopes,
#       OR the MCP feature isn't toggled ON in the app settings.
#
#  ❌ 403 Forbidden (Notion)
#     → The integration hasn't been connected to that page/database.
#       Open the Notion page → ··· → Connections → Add integration.
#
#  ❌ Connection timeout / no tools returned
#     → Check the URL is correct and the server is reachable.
#       Try curl -X POST <url> to test directly.
#
#  ❌ tool_name conflict (two tools named "search")
#     → Already handled: tool_name_prefix=True renames them to
#       "github__search", "linear__search", etc.
#
# =============================================================================
