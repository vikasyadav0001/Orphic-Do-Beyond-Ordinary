from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from document_parser.doc_parser_rag import get_or_create_vector_store, asearch_documents
from utils.logger import get_logger
from config import get_settings

logger = get_logger(__name__)
env = get_settings()

@tool
async def search_uploaded_documents(query: str, file_path: str, config: RunnableConfig) -> str:
    """
    Use this tool to search uploaded documents (PDFs, CSVs, Text files, Code) for information.
    Pass a specific search query and the relevant file path of the document.
    """
    try:
        user_id = config.get("configurable", {}).get("user_id", "default_user")
        session_id = config.get("configurable", {}).get("thread_id", "test-thread-3")

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
    import ast
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, SystemMessage

    # ── 1. Load the file ──────────────────────────────────────────────────────
    if file_path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)

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
        model="meta-llama/llama-4-scout-17b-16e-instruct",
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

    # ── 4. Validate: only allow safe AST node types ───────────────────────────
    BLOCKED_NODES = {
        ast.Import, ast.ImportFrom,   # no imports
        ast.Delete,                   # no del statements
    }
    BLOCKED_CALLS = {"eval", "exec", "open", "compile", "__import__", "breakpoint"}

    try:
        tree = ast.parse(generated_code)
    except SyntaxError as e:
        return f"[compute_on_csv] LLM returned unparseable code: {e}\n\nCode:\n{generated_code}"

    for node in ast.walk(tree):
        if type(node) in BLOCKED_NODES:
            return "[compute_on_csv] Blocked: generated code contains disallowed statements."
        if isinstance(node, ast.Call):
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            if func_name in BLOCKED_CALLS:
                return f"[compute_on_csv] Blocked: disallowed function call '{func_name}'."

    # ── 5. Execute in a restricted namespace ─────────────────────────────────
    namespace: dict = {"df": df, "pd": pd}
    try:
        exec(generated_code, namespace)  # noqa: S102
    except Exception as e:
        return (
            f"[compute_on_csv] Execution error: {e}\n\n"
            f"Generated code:\n{generated_code}"
        )

    result = namespace.get("result", None)

    if result is None:
        return (
            "[compute_on_csv] Code ran but 'result' was never set.\n\n"
            f"Generated code:\n{generated_code}"
        )

    # ── 6. Serialize result to a clean string ─────────────────────────────────
    if isinstance(result, pd.DataFrame):
        return result.to_string(index=False)
    if isinstance(result, pd.Series):
        return result.to_string()
    return str(result)





custom_tools = [search_uploaded_documents, extract_full_text, compute_on_csv_or_excel]