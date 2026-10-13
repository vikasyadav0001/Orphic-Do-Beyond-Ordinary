import httpx
import json
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from document_parser.doc_parser_rag import get_or_create_vector_store, asearch_documents
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphInterrupt, GraphRecursionError
from langgraph.types import interrupt
from services.connections.oauth_engine import OAuthEngine
from services.connections.providers import PROVIDERS
from utils.agent_helpers import resolve_token
from utils.logger import get_logger
from config import get_settings

logger = get_logger(__name__)
env = get_settings()
oauth_engine = OAuthEngine()

@tool
async def search_uploaded_documents(query: str, file_path: str, config: RunnableConfig) -> str:
    """
    Use this tool to search uploaded documents (PDFs, CSVs, Text files, Code) for information.
    Pass a specific search query and the relevant file path of the document.
    """
    try:
        user_id = config.get("configurable", {}).get("user_id")
        session_id = config.get("configurable", {}).get("thread_id")

        if not user_id or not session_id:
            return "Provide proper user_id and session_id, unable to search documents without them."

        logger.info(f"Triggered Document search for user: {user_id}")

        import asyncio
        vector_store = await asyncio.to_thread(
            get_or_create_vector_store, file_path, session_id, user_id
        )
        # Filter by source file so search is scoped to this document only,
        # not the user's entire collection across all sessions.
        context = await asearch_documents(
            vector_store,
            query,
            filter={"source": file_path},
        )

        if not context:
            return "No relevant information found in the documents."
            
        return "\n\n".join(context)

    except Exception as e:
        logger.error(f"Error during Document search: {e}", exc_info=True)
        return f"Failed to search the document for the user: {user_id}, Error: {e}"
        

@tool
async def extract_full_text(file_path: str, pages: str = "all") -> str:
    """
    Use this when the user wants the raw text content of a document.
    Use search_uploaded_documents instead when they have a specific question.
    pages: "all", "1-3", "first", "last"
    """
    import fitz  
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text[:8000]


@tool
async def compute_on_csv_or_excel(file_path: str, question: str) -> str:
    """
    Use this for ANY question that requires aggregation, counting, filtering,
    or computation on a CSV/Excel file. Examples:
    - "What is the total of column X?"
    - "How many rows have value > 100?"
    - "What is the average of Y grouped by Z?"
    """
    import pandas as pd
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, SystemMessage
    import subprocess
    import tempfile
    import os

    # ── 1. Load the file ──────────────────────────────────────────────────────
    try:
        if file_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
    except Exception as e:
        return f"[Error]: Failed to read the file for schema: {e}"

    # ── 2. Build a compact schema description ─────────────────────────────────
    schema_lines = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample = df[col].dropna().head(3).tolist()
        schema_lines.append(f"  - {col!r} ({dtype}): sample values → {sample}")

    schema_str = "\n".join(schema_lines)
    shape_str = f"{len(df)} rows × {len(df.columns)} columns"

    # ── 3. Ask the LLM to generate pandas code ────────────────────────────────
    system_prompt = """You are a pandas code generator.
Given a DataFrame `df` (already loaded) and a user question, write Python code to answer it.

Rules:
- The DataFrame is already in scope as `df`.
- Store the final answer in a variable called `result`.
- `result` must be a string, number, or something easily str()-convertible.
- Do NOT import pandas or read any file — df is already loaded.
- Return ONLY the raw Python code block, no markdown fences, no explanation.
"""

    user_prompt = f"""DataFrame shape: {shape_str}

Columns:
{schema_str}

Question: {question}

Write the pandas code now:"""

    client = ChatGroq(
        model="qwen/qwen3.6-27b",
        api_key=env.groq_api_key
    )

    messages=[
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    response = await client.ainvoke(messages)
    generated_code = response.content.strip()

    # Strip markdown fences if the model adds them anyway
    if generated_code.startswith("```"):
        generated_code = "\n".join(
            line for line in generated_code.splitlines()
            if not line.strip().startswith("```")
        ).strip()
        if generated_code.endswith("```"):
            generated_code = generated_code[:-3].strip()

    with tempfile.NamedTemporaryFile(mode='w', suffix='py', delete=False) as  temp_script:
        temp_script.write(generated_code)
        script_path = temp_script.name

        try:
            process= subprocess.run(
                ["python3", script_path],
                capture_output=True,
                text=True,
                timeout=10
            )

            if process.returncode != 0:
                return (
                f"[compute_on_csv] Script failed with error:\n{process.stderr}\n\n"
                f"Generated code:\n{generated_code}"
            )

            result = process.stdout.strip()
            if not result:
                return "[compute_on_csv] Code ran successfully but printed nothing"
            
            return result
        
        except subprocess.TimeoutExpired:
            return "[compute_on_csv] Error: The generated code took too long to execute and was terminated (infinite loop or heavy computation)."
        except Exception as e:
            return f"[compute_on_csv] Unexpected execution error: {e}"
        finally:
            if os.path.exists(script_path):
                os.remove(script_path)

#helper function for workflow agent
def _extract_n8n_workflow_id(messages: list) -> str | None:
    import json
    for msg in reversed(messages):
        if hasattr(msg, "name") and msg.name in ("create_workflow_from_code", "publish_workflow"):
            try:
                data = json.loads(msg.content)
                # n8n returns {"id": "...", "name": "...", ...}
                if "id" in data:
                    return str(data["id"])
            except Exception:
                pass
    return None

async def _upsert_workflow(user_id, n8n_workflow_id, name, providers, source_conversation_id):
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from db.models import get_async_session, Workflows
    from  sqlalchemy import func

    async with get_async_session() as session:
        stmt = pg_insert(Workflows).values(
            user_id=user_id,
            n8n_workflow_id=n8n_workflow_id,
            name=name,
            providers=providers,
            source_conversation_id=source_conversation_id,
        ).on_conflict_do_update(
            index_elements=["n8n_workflow_id"],
            set_={"name": name, "providers": providers, "is_active": True, "updated_at": func.now()}
        )
        await session.execute(stmt)
        await session.commit()


@tool
async def n8n_workflow_agent(
    task_description: str,
    providers: list[str] = [],
    config: RunnableConfig = None,
) -> str:
    """
    Delegate workflow automation and n8n management tasks to a specialized N8N Assistant Agent.
    Use this for ANY operations involving n8n, workflows, credentials, or executions, including:
    - Creating, validating, testing, and publishing/activating new automation workflows.
    - Searching, listing, updating, archiving, or checking execution logs/details of existing workflows.
    - Listing credentials, checking if they exist, or creating new credentials.
    - Managing n8n data tables, projects, and folders.
    
    task_description: The natural language request detailing what the workflow assistant should do.
                      Examples: 
                      - "Search for my Slack workflows and tell me if they are active"
                      - "Check the last execution status of workflow ID 12"
                      - "Build a workflow that posts GitHub PR summaries to Slack"
                      - "List all registered credentials"
    providers: Optional list of provider names (e.g. ['github', 'slack']) if the query requires 
               establishing or verifying OAuth connections before proceeding.
    """

    from agents.workflow_agent import get_workflow_agent

    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "[Error] Could not identify user. Cannot create workflow."

    if not task_description or not task_description.strip():
        return "[Error] task_description is required. Describe what the workflow should do."

    _workflow_agent = await get_workflow_agent()

    if _workflow_agent is None:
        return (
            "Workflow automation is not available right now. "
            "The n8n service may not be configured yet. "
            "Please contact support or try again later."
        )

    # ── Sequential auth check for all required providers ─────────────────────
    # One interrupt per missing provider.
    # GraphInterrupt propagates to compiled_graph — frontend shows connect button.
    # On resume, execution continues from the interrupt() return value.
    for provider in providers:
        if provider not in PROVIDERS:
            return (
                f"[Error] Unknown provider '{provider}'. "
                f"Supported providers: {', '.join(PROVIDERS.keys())}"
            )

        token = await resolve_token(user_id, provider)

        if token is None:
            try:
                auth_url = await oauth_engine.get_authorization_url(provider, user_id)
            except Exception as e:
                logger.error(
                    f"Auth URL generation failed for {provider}/{user_id}: {e}"
                )
                auth_url = ""

            from utils.interrupts import read_resume_value
            
            resume_val = interrupt({
                "id": f"auth_wf_agent_{provider}",
                "kind": "connector_required",
                "type": "interrupt",
                "interrupt_type": "auth_required",
                "provider": PROVIDERS[provider].label,
                "provider_name": provider,
                "auth_url": auth_url,
                "message": f"Connect {PROVIDERS[provider].label} to create this workflow.",
                "actions": ["connected", "cancel"]
            })

            decision, user_input = read_resume_value(resume_val)

            if decision in {"cancel", "reject", "skip"}:
                return "Workflow creation cancelled."

            # Re-check after resume
            token = await resolve_token(user_id, provider)
            if token is None:
                return (
                    f"[Error] Still not connected to {PROVIDERS[provider].label}. "
                    f"Workflow creation aborted. "
                    f"Please connect via Connectors and try again."
                )

    # ── All providers authenticated — invoke Deep Agent ───────────────────────
    logger.info(
        f"Invoking workflow agent for user {user_id}: "
        f"providers={providers} task='{task_description[:80]}'"
    )

    try:
        import asyncio
        async with asyncio.timeout(120):
            result = await _workflow_agent.ainvoke(
                {"messages": [HumanMessage(content=task_description)]},
                config={**config, "recursion_limit": 50},
                version="v2",
            )

        n8n_workflow_id = _extract_n8n_workflow_id(result.get("messages", []))

        if n8n_workflow_id:
            await _upsert_workflow(
                user_id=user_id,
                n8n_workflow_id=n8n_workflow_id,
                name=task_description[:100],
                providers=providers,
                source_conversation_id=config["configurable"].get("thread_id"),
            )

        return result["messages"][-1].content

    except asyncio.TimeoutError:
        logger.error("[WF] Workflow agent timed out after 120s")
        return "Workflow creation timed out. Please try again with a simpler description."
    
    except GraphRecursionError:
        logger.error("[WF] Recursion limit hit — see tool calls above for loop cause")
        raise

    except GraphInterrupt:
        # publish_workflow HITL or nested auth interrupt from create_n8n_credential
        # Must reach compiled_graph — never catch and swallow
        raise

    except Exception as e:
        logger.error(
            f"Workflow creation failed for user {user_id}: "
            f"{type(e).__name__}: {e}",
            exc_info=True,
        )
        return (
            f"Workflow creation failed: {str(e)}. "
            f"Please try again or simplify the workflow description."
        )

@tool
async def send_email(
    to: str,
    subject: str,
    body: str,
    config: RunnableConfig
) -> str:
    """Send an email via Gmail."""
    user_id = config["configurable"]["user_id"]
    token = await resolve_token(user_id, "google_gmail")
    if not token:
        return "User is not Connected to Gmail, Ask user to Authenticate with Gmail."
    
    import base64
    from email.mime.text import MIMEText
    
    msg = MIMEText(body)
    msg["to"] = to
    msg["subject"] = subject
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            headers={"Authorization": f"Bearer {token}"},
            json={"raw": raw}
        )
        r.raise_for_status()
        return f"Email sent. Message ID: {r.json()['id']}"

@tool
async def list_emails(
    max_results: int = 10,
    query: str = "",
    config: RunnableConfig = None
) -> str:
    """List recent emails from Gmail inbox."""
    user_id = config["configurable"]["user_id"]
    token = await resolve_token(user_id, "google_gmail")
    if not token:
        return "User unauthenticated with Gmail, ask user to connect first with gmail"

    params = {"maxResults": max_results, "labelIds": "INBOX"}
    if query:
        params["q"] = query

    async with httpx.AsyncClient() as client:
        # Step 1: list message IDs
        r = await client.get(
            "https://gmail.googleapis.com/gmail/v1/users/me/messages",
            headers={"Authorization": f"Bearer {token}"},
            params=params
        )
        r.raise_for_status()
        messages = r.json().get("messages", [])

        # Step 2: fetch snippet + headers for each
        results = []
        for msg in messages:
            r2 = await client.get(
                f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{msg['id']}",
                headers={"Authorization": f"Bearer {token}"},
                params={"format": "metadata", "metadataHeaders": ["From", "Subject", "Date"]}
            )
            r2.raise_for_status()
            data = r2.json()
            headers = {h["name"]: h["value"] for h in data["payload"]["headers"]}
            results.append({
                "id": msg["id"],
                "from": headers.get("From"),
                "subject": headers.get("Subject"),
                "date": headers.get("Date"),
                "snippet": data.get("snippet")
            })

    return json.dumps(results, indent=2)

from tools.search_tools import search_tools

custom_tools = [send_email, list_emails, search_uploaded_documents, extract_full_text, compute_on_csv_or_excel, search_tools]

