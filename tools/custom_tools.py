from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from document_parser.doc_parser_rag import get_or_create_vector_store, asearch_documents
from utils.logger import get_logger


logger = get_logger(__name__)


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
        
        vector_store = get_or_create_vector_store(file_path, session_id=session_id, user_id=user_id)
        context = await asearch_documents(vector_store, query)

        if not context:
            return "No relevant information found in the documents."
            
        return "\n\n".join(context)

    except Exception as e:
        logger.error(f"Error during Document search: {e}", exc_info=True)
        return f"Failed to search the document for the user: {user_id}, Error: {e}"
        

custom_tools = [search_uploaded_documents]