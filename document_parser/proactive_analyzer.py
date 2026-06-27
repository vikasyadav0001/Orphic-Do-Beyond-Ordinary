import os
import pandas as pd
from pymupdf import open as pdf_open
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from .doc_parser_rag import compute_hash, parse_document, get_or_create_vector_store, asearch_documents
from utils.logger import get_logger
from config import get_settings

logger = get_logger(__name__)

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=get_settings().groq_api_key,
    temperature=0.7
)

SYSTEM_PROMPT = """You are an intelligent document assistant. 
A user has just uploaded a document but hasn't asked a question yet.
Based on the preview provided, you must:
1. Identify the document type (research paper, report, code, data, etc.)
2. Identify 3-4 key topics or themes
3. Offer 2-3 specific, concrete ways you can help (reference actual content you see)
Keep your response SHORT and conversational (2-3 sentences max).
Reference specific details from the preview to make it feel personalized.
{}
Example:
"I see a 24-page research paper on transformer architectures. I can summarize the key findings, 
explain the methodology in section 3, or compare the benchmark results. What would help most?"
"""

def extract_preview(file_path: str, max_tokens:int = 1000) -> str:
    """
    Extract a lightweight preview of a document for the proactive analyzer.
    Just enough to understand what it is, not the full content.
    """
    logger.info(f"Extracting preview from {file_path}")
    ext =  file_path.lower().split(".")[-1]

    try:
        if ext == "pdf":
            doc = pdf_open(file_path)
            preview_text = ""
            for page_num in range(min(2, len(doc))):
                page = doc[page_num]
                preview_text += page.get_text()
            doc.close()
        
        elif ext == "csv":
            df = pd.read_csv(file_path)
            preview_text = f"Columns: {df.columns.tolist()}\n{df.head(10).to_string()}"

        elif ext in ["xlsx", "xls"]:
            df = pd.read_excel(file_path)
            preview_text = f"Columns: {df.columns.tolist()}\n{df.head(10).to_string()}"

        elif ext in ['txt', 'md', 'py', 'json', 'js', 'html', 'css']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                preview_text = f.read(max_tokens * 4)
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                preview_text = f.read(max_tokens * 4)
        
        preview_text = preview_text[: max_tokens * 4]
        logger.info(f"Preview extracted: {len(preview_text)} chars")
        return preview_text
        
    except Exception as e:
        logger.error(f"Failed to extract preview: {e}", exc_info=True)
        return ""


async def stream_opening_offer(preview_text: str, file_name: str):
    """
    Async generator that streams opening offer tokens in real-time.
    """
    logger.info(f"Streaming opening offer for {file_name}")
    message = f"Document preview:\n\n{preview_text}\n\nFile name: {file_name}"
    
    try:
        async for chunk in llm.astream(
            [SystemMessage(content=SYSTEM_PROMPT.format(message))]
        ):
            if chunk and chunk.content:
                yield chunk.content
    except Exception as e:
        logger.error(f"Failed to stream opening offer: {e}", exc_info=True)
        yield f"I've loaded {file_name}. What would you like to know about it?"


async def generate_opening_offer(preview_text: str, file_name: str) -> str:
    """
    Static/non-streaming fallback. Collects the streamed tokens into a single string.
    Keeps the Document Ingestion Graph fully functional without any changes.
    """
    tokens = []
    async for token in stream_opening_offer(preview_text, file_name):
        tokens.append(token)
    return "".join(tokens)
