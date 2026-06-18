import os
import fitz
import hashlib
from datetime import datetime
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_community.document_loaders import CSVLoader, TextLoader, UnstructuredExcelLoader
from langchain_groq import ChatGroq
from config import get_settings
from typing import List
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from utils.logger import get_logger

logger = get_logger(__name__)

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=get_settings().groq_api_key,
)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1536,
    api_key=get_settings().openai_api_key
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)


def compute_hash(file_path: str) -> str:
    """Compute SHA256 hash of file contents for deduplication."""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.error(f"Failed to compute hash for {file_path}: {e}")
        raise


def parse_pdf(file_path: str) -> List[Document]:
    logger.info(f"Parsing the Image based documents at {file_path}")

    doc = fitz.open(file_path)
    page_count = len(doc)

    try:
        image_llm = LLMImageBlobParser(
            model=llm,
            prompt="Describe this image in detail, including any text, charts, data, diagrams, or visual information."
        )
        parser = PyMuPDF4LLMLoader(
            file_path=file_path,
            extract_images=True,
            images_parser=image_llm
        )
        docs = parser.load()
        logger.info("Docs parsed successfully.")
        return docs
    except Exception as e:
        logger.error(f"Failed to parse PDF: {e}", exc_info=True)
        raise

# ← UPDATED: Added user_id parameter
def get_vector_store(user_id: str) -> PGVector:
    """Helper to get the PGVector instance for a specific user (merged collection)."""
    return PGVector(
        embeddings=embedding_model,
        connection=get_settings().db_url,
        collection_name=f"user_{user_id}_docs",  
    )

def parse_document(file_path: str) -> List[Document]:
    """Universal document parser that selects the correct loader based on file extension."""
    ext = file_path.lower().split('.')[-1]
    
    if ext == 'pdf':
        return parse_pdf(file_path)
    elif ext == 'csv':
        logger.info(f"Parsing CSV file: {file_path}")
        loader = CSVLoader(file_path=file_path)
        return loader.load()
    elif ext in ["xlsx", "xls"]:
        loader = UnstructuredExcelLoader(file_path=file_path, mode="elements")
        return loader.load()
    elif ext in ['txt', 'md', 'py', 'json', 'js', 'html', 'css']:
        logger.info(f"Parsing Text/Code file: {file_path}")
        loader = TextLoader(file_path=file_path)
        return loader.load()
    else:
        logger.warning(f"Unsupported file type: {ext}. Attempting fallback text parsing...")
        loader = TextLoader(file_path=file_path)
        return loader.load()

# ← UPDATED: Added user_id, hash-based dedup, enhanced metadata
def get_or_create_vector_store(file_path: str, session_id: str, user_id: str) -> PGVector:
    """
    Returns existing PGVector or creates new one if file doesn't exist.
    Strategy 2: One collection per user, deduped by doc_hash.
    """
    try:
        # 1. Connect to the merged collection for this user
        vector_store = get_vector_store(user_id)
        
        # 2. Compute file hash for deduplication
        file_hash = compute_hash(file_path)
        logger.info(f"File hash: {file_hash}")
        
        # 3. Check if this exact file is already in the user's collection
        # Filter by doc_hash to find duplicates
        try:
            existing_docs = vector_store.similarity_search(
                query="",  # dummy query, we're just checking metadata
                k=1,
                filter={"doc_hash": file_hash}
            )
            
            if existing_docs and len(existing_docs) > 0:
                logger.info(f"File hash {file_hash} already indexed for user {user_id}. Skipping re-ingestion!")
                return vector_store
        except Exception as e:
            logger.warning(f"Filter search failed (may be first doc): {e}. Proceeding with ingestion.")
        
        # 4. File is new, parse and ingest
        logger.info(f"File not found in user {user_id}'s collection. Parsing now...")
        docs = parse_document(file_path)
        
        # 5. Attach rich metadata to all documents (Strategy 2)
        file_ext = file_path.lower().split('.')[-1]
        for doc in docs:
            doc.metadata = {
                "source": file_path,
                "doc_hash": file_hash,                        # ← Dedup key
                "session_id": session_id,                     # ← Session context
                "upload_time": datetime.now().isoformat(),    # ← Freshness
                "doc_type": file_ext,                         # ← File type
                "user_id": user_id                            # ← User scope
            }
        
        # 6. Split and ingest
        logger.info("Ingesting document into PGVector db...")
        splitted_docs = splitter.split_documents(docs)
        vector_store.add_documents(splitted_docs)
        logger.info(f"PGVector ready. {len(splitted_docs)} chunks stored for user {user_id}.")
        
        return vector_store
        
    except Exception as e:
        logger.error(f"Failed to get or create vector store: {e}", exc_info=True)
        raise

async def asearch_documents(vector_store: PGVector, query: str, k=3) -> List[str]:
    """Asynchronous similarity search for LangGraph."""
    logger.info("Performing similarity search...")
    try:
        retrieved_docs = await vector_store.asimilarity_search(query=query, k=k)
        context = [d.page_content for d in retrieved_docs]
        logger.info(f"Retrieved {len(context)} documents.")
        return context
    except Exception as e:
        logger.error(f"Similarity search failed: {e}", exc_info=True)
        raise

# ← UPDATED: Added user_id parameter
async def main(query: str, file_path: str, session_id: str, user_id: str) -> List[str]:
    """Main RAG pipeline entrypoint."""
    logger.info("Initializing RAG pipeline...")
    vector_store = get_or_create_vector_store(file_path, session_id, user_id)
    context = await asearch_documents(vector_store, query)
    return context

if __name__ == "__main__":
    import asyncio
    
    # Example call with all required params
    file_path = os.path.join(os.path.dirname(__file__), "somatosensory.pdf")
    query = "What is Somatosensory System?"
    session_id = "session_demo_001"
    user_id = "vikas"
    
    text = asyncio.run(main(query=query, file_path=file_path, session_id=session_id, user_id=user_id))
    print(f"Here is the retrieved text: \n{text}")