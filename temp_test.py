#file for temporary testing scripts
from pathlib import Path
from typing import List
import fitz  # Core PyMuPDF to check page length
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=get_settings().groq_api_key,
)

image_llm = LLMImageBlobParser(
    model=llm,
    prompt="Describe this image in detail, including any text, charts, data, diagrams, or visual information."
)

def parse_pdf_explicit(file_path: str) -> List[Document]:
    try:
        with fitz.open(file_path) as doc:
            page_count = len(doc)

        if page_count == 0:
            logger.error("0 pages explicitly detected.")
            return [Document(page_content="[Error: Empty PDF]", metadata={"source": file_path})]

        if page_count == 1:
            logger.info("Single page document. Using single mode without image extraction.")
            parser = PyMuPDF4LLMLoader(
                file_path=file_path,
                mode="single",
                extract_images=True,
            )
            return parser.load()

        # Multi-page: let the loader handle the full doc but without image extraction
        # to avoid the internal image map index bug
        logger.info(f"Multi-page document ({page_count} pages). Running full document parse.")
        parser = PyMuPDF4LLMLoader(
            file_path=file_path,
            mode="page",
            extract_images=True
        )
        return parser.load()

    except Exception as e:
        logger.error(f"Loader failed: {e}")
        raise



if __name__ == "__main__":
    file_path = "/home/dell/Desktop/Orphic-Do-Beyond-Ordinary/uploads/AI Engineering Training Plan_ From Fundamentals to Production _ Claude.pdf"
    result = parse_pdf_explicit(file_path)
    print(result)