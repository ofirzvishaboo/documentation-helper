import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
import dotenv
from langchain_tavily import TavilyCrawl, TavilyMap, TavilyExtract
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_pinecone import PineconeVectorStore
from logger import (Colors, log_info, log_success, log_error, log_warning, log_header)

dotenv.load_dotenv()


ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", show_progress_bar=False, chunk_size=50, retry_min_seconds=10
)

# chroma = Chroma(persistence_directory="chroma_db", embedding_function=embeddings)
vectorstore = PineconeVectorStore(index_name="langchain-doc-index", embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def main():
    """Main async function to the entire program."""
    log_header("Documentation Ingestion Pipeline")

    log_info(
        "Tavily Crawl: Starting to crawl documentation from https://python.langchain.com",
        Colors.PURPLE,
    )

    res = tavily_crawl.invoke({
        "url": "https://python.langchain.com",
        "max_depth": 5,
        "extract_depth": "advanced",
        "instructions": "content on ai agents",
    })

    all_docs = [Document(page_content=result["raw_content"], metadata={"source": result["url"]}) for result in res["results"]]

    log_success(
        f"Tavily Crawl: Successfully crawled {len(all_docs)} urls from documentation site",
    )


if __name__ == "__main__":
    asyncio.run(main())
