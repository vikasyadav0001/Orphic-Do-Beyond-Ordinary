"""
=============================
Custom tool used exclusively by the workflow Deep Agent.
Fetches the user's stored OAuth token for a provider
and registers it as a credential in n8n via REST API.

NOT in the main agent's tool list — only bound to the workflow agent.
"""

import httpx
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.types import interrupt

from config import get_settings
from services.connections.oauth_engine import OAuthEngine
from services.connections.providers import PROVIDERS
from utils.agent_helpers import resolve_token
from utils.logger import get_logger
from utils.interrupts import read_resume_value

settings = get_settings()
logger = get_logger(__name__)
oauth_engine = OAuthEngine()

# Maps Orphic provider names → n8n credential types
# Extend this as new providers are added to PROVIDERS
_N8N_CREDENTIAL_TYPE_MAP: dict[str, str] = {
    "github": "githubOAuth2Api",
    "google_gmail": "googleOAuth2Api",
    "google_drive": "googleOAuth2Api",
    "google_sheets": "googleOAuth2Api",
    "google_docs": "googleOAuth2Api",
    "google_calendar": "googleOAuth2Api",
    "notion": "mcpOAuth2Api",
    "slack": "slackOAuth2Api",
    "atlassian": "mcpOAuth2Api"
}

def _build_credential_data(provider: str, token: str) -> dict:
    if provider == "notion":
        return {"apiKey": token}
    elif provider == "github":
        return {
            "server": "https://api.github.com",
            "user": "",          # n8n requires it, empty is fine for PAT auth
            "accessToken": token
        }
    elif provider == "slack":
        return {"accessToken": token}
    elif provider == "atlassian":
        return {"apiKey": token}  # also needs email + domain but those aren't in token
    elif provider.startswith("google_"):
        # googleOAuth2Api can't be injected via REST — skip for now
        return None
    else:
        return {"accessToken": token}  # safe default

@tool
async def create_n8n_credential(provider: str, config: RunnableConfig) -> str:
    """
    Fetch the user's stored OAuth token for a provider and register it
    as a credential in n8n so workflows can authenticate with that service.

    Always call list_credentials first to check if the credential already
    exists — only call this if it is missing.

    provider: the provider name (e.g. 'github', 'google', 'notion', 'slack').
    """
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "[Error] Could not identify user. Cannot create credential."

    if provider not in PROVIDERS:
        return (
            f"[Error] Unknown provider '{provider}'. "
            f"Supported: {', '.join(PROVIDERS.keys())}"
        )
    

    n8n_token = settings.n8n_token
    if not n8n_token:
        return "[Error] N8N_TOKEN not configured. Cannot create credential."

    n8n_base_url = getattr(settings, "n8n_base_url", "http://localhost:5678")
    provider_label = PROVIDERS[provider].label

    # ── Resolve token — interrupt if not authenticated ────────────────────────
    # GraphInterrupt propagates up through workflow_tool → tool_node → graph
    # Frontend shows connect button, user authenticates, resumes here
    token = await resolve_token(user_id, provider)

    if token is None:
        try:
            auth_url = await oauth_engine.get_authorization_url(provider, user_id)
        except Exception as e:
            logger.error(f"Auth URL generation failed for {provider}/{user_id}: {e}")
            auth_url = ""
        
        resume_val = interrupt({
            "id": f"auth_wf_{provider}",
            "kind": "connector_required",
            "type": "interrupt",
            "interrupt_type": "auth_required",
            "provider": provider_label,
            "provider_name": provider,
            "auth_url": auth_url,
            "message": f"Connect {provider_label} to continue building this workflow.",
            "actions": ["connected", "cancel"]
        })

        decision, user_input = read_resume_value(resume_val)

        if decision in {"cancel", "reject", "skip"}:
            return f"Credential creation for {provider_label} cancelled."

        # Re-check after user authenticates and resumes
        token = await resolve_token(user_id, provider)
        if token is None:
            return (
                f"[Error] Still not connected to {provider_label} after authentication. "
                f"Please try connecting again via Settings."
            )
    credential_type = _N8N_CREDENTIAL_TYPE_MAP.get(provider)
    if not credential_type:
        return (
            f"[Error] No n8n credential type mapped for '{provider}'. "
            f"Do not retry. Contact support to add this provider."
        )

    credential_data = _build_credential_data(provider, token)
    if not credential_data:
        return (
        f"[Error] Credential injection not supported for {provider_label}. "
        f"Do not retry. Proceed without this credential or abort."
        )
    # ── Register credential in n8n ────────────────────────────────────────────
    payload = {
        "name": f"orphic_{provider}_{user_id[:8]}",
        "type": credential_type,
        "data": credential_data,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            existing = await client.get(
                f"{n8n_base_url}/api/v1/credentials",
                headers={"X-N8N-API-KEY": n8n_token},
            )
            if existing.status_code == 200:
                creds = existing.json().get("data", [])
                cred_name = f"orphic_{provider}_{user_id[:8]}"
                for cred in creds:
                    if cred.get("name") == cred_name:
                        logger.info(f"Credential already exists for {provider}, reusing id={cred['id']}")
                        return f"Credential for {provider_label} already exists (id: {cred['id']}). Reuse it."
            else:
                logger.warning(f"Could not list credentials ({existing.status_code}), proceeding with creation")

            response = await client.post(
                f"{n8n_base_url}/api/v1/credentials",
                json=payload,
                headers={
                    "X-N8N-API-KEY": n8n_token,
                    "Content-Type": "application/json",
                },
            )

        if response.status_code in (200, 201):
            credential_id = response.json().get("id", "unknown")
            logger.info(
                f"Created n8n credential for {provider}/{user_id}: id={credential_id}"
            )
            return (
                f"Successfully created {provider_label} credential in n8n "
                f"(id: {credential_id}). You can now use it in workflows."
            )

        # n8n returned an error
        logger.error(
            f"n8n credential creation failed for {provider}/{user_id}: "
            f"{response.status_code} {response.text}"
        )
        return (
            f"[Error] n8n rejected the credential for {provider_label}: "
            f"{response.status_code} — {response.text[:200]}",
            f"Do not retry this call. Report this error to the user and stop."
        )

    except httpx.TimeoutException:
        logger.error(f"n8n timeout creating credential for {provider}/{user_id}")
        return (
            "[Error] n8n did not respond in time. "
            "Is the n8n service running on the VM?"
        )
    except Exception as e:
        logger.error(
            f"create_n8n_credential unexpected error for {provider}/{user_id}: "
            f"{type(e).__name__}: {e}",
            exc_info=True,
        )
        return f"[Error] Credential creation failed unexpectedly: {str(e)}"