import os
import re
import logging
from typing import List, Dict, Any
from pathlib import Path

import fitz
import wikipediaapi
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vector_store import get_pinecone_client, get_or_create_index, upsert_chunks
from snowflake_logger import PipelineRunLogger

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# --- Configuration Constants ---
TOPICS = [
    "Data engineering", "Extract, transform, load", "Data pipeline",
    "Snowflake Inc", "Apache Airflow", "Retrieval-augmented generation",
    "Large language model", "Vector database", "Embedding (machine learning)",
    "Python (programming language)", "Data warehouse", "Data lake",
    "Apache Spark", "Apache Kafka", "Distributed computing", "MLOps",
    "Prompt engineering", "Transformer (deep learning)", "Semantic search",
    "Knowledge graph", "CI/CD", "Data governance", "Data mesh",
    "Tokenization", "Graph database"
]

BOOKS = [
    {
        "blob_name": "fundamentals_of_data_engineering.pdf",
        "title": "Fundamentals of Data Engineering",
        "url": "https://freecomputerbooks.com/books/Fundamentals-of-Data-Engineering.pdf",
    },
    {
        "blob_name": "The-Data-Engineers-Guide-to-Apache-Spark.pdf",
        "title": "The Data Engineer's Guide to Apache Spark",
        "url": "https://github.com/xrenaissance/Functional-Programming_in_Scala_Specialization/blob/master/The-Data-Engineers-Guide-to-Apache-Spark.pdf",
    },
    {
        "blob_name": "Generative-AI-and-LLMs-for-Dummies.pdf",
        "title": "Generative AI and LLMs for Dummies",
        "url": "https://www.snowflake.com/wp-content/uploads/2024/01/Generative-AI-and-LLMs-for-Dummies.pdf",
    },
    {
        "blob_name": "cloud-data-engineering-for-dummies-.pdf",
        "title": "Cloud Data Engineering for Dummies",
        "url": "https://www.snowflake.com/wp-content/uploads/2020/12/cloud-data-engineering-for-dummies-.pdf",
    },
    {
        "blob_name": "Cloud-Data-Warehousing-For-Dummies-3rd-Edition.pdf",
        "title": "Cloud Data Warehousing for Dummies",
        "url": "https://www.snowflake.com/wp-content/uploads/2023/11/Cloud-Data-Warehousing-For-Dummies-3rd-Edition.pdf",
    },
    {
        "blob_name": "Building-Applications-with-Snowpark-for-Dummies.pdf",
        "title": "Building Applications with Snowpark for Dummies",
        "url": "https://www.snowflake.com/wp-content/uploads/2024/01/Building-Applications-with-Snowpark-for-Dummies.pdf",
    },
]

CHUNK_SIZE    = 800
CHUNK_OVERLAP = 80
EMBED_MODEL   = "all-MiniLM-L6-v2"
EMBED_BATCH   = 256


def clean_text(text: str) -> str:
    """Standardise and clean extracted text before embedding."""
    text = re.sub(r"\.{3,}", " ", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_wikipedia_articles() -> List[Dict[str, Any]]:
    """Fetch Wikipedia articles for all configured topics."""
    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="DataEngineeringAssistant/1.0 (contact: your@email.com)",
    )
    articles = []
    for topic in TOPICS:
        try:
            page = wiki.page(topic)
            if page.exists():
                articles.append({
                    "title":       page.title,
                    "url":         page.fullurl,
                    "text":        page.text,
                    "source_type": "wikipedia",
                })
        except Exception as e:
            logger.error(f"Failed to fetch Wikipedia topic '{topic}': {e}")

    logger.info(f"Fetched {len(articles)} Wikipedia articles.")
    return articles


def ingest_azure_pdfs() -> List[Dict[str, Any]]:
    """Download PDFs from Azure Blob Storage and extract their text."""
    connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container   = os.getenv("AZURE_CONTAINER_NAME")

    if not connect_str or not container:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING and AZURE_CONTAINER_NAME must be set.")

    blob_service = BlobServiceClient.from_connection_string(connect_str)
    book_data    = []

    for book in BOOKS:
        try:
            logger.info(f"Processing: {book['title']}")
            blob_client = blob_service.get_blob_client(
                container=container,
                blob=book["blob_name"],
            )
            pdf_bytes      = blob_client.download_blob().readall()
            doc            = fitz.open(stream=pdf_bytes, filetype="pdf")
            full_book_text = "".join([page.get_text() for page in doc])

            book_data.append({
                "title":       book["title"],
                "url":         book["url"],
                "text":        clean_text(full_book_text),
                "source_type": "book",
            })
        except Exception as e:
            logger.error(f"Failed to process '{book['title']}': {e}")

    return book_data


def chunk_documents(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Split documents into overlapping chunks for semantic indexing."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    all_chunks = []
    for doc in docs:
        for i, chunk_text in enumerate(splitter.split_text(doc["text"])):
            all_chunks.append({
                "id":          f"{doc['source_type']}_{doc['title']}_{i}",
                "title":       doc["title"],
                "url":         doc["url"],
                "text":        chunk_text,
                "source_type": doc["source_type"],
                "chunk_index": i,
            })

    logger.info(f"Created {len(all_chunks)} chunks from {len(docs)} documents.")
    return all_chunks


def embed_and_sync(chunks: List[Dict[str, Any]]) -> int:
    """Generate embeddings in batches and upsert them to Pinecone."""
    embed_model = SentenceTransformer(EMBED_MODEL)
    pc          = get_pinecone_client()
    index_name  = os.getenv("PINECONE_INDEX_NAME")
    index       = get_or_create_index(pc, index_name)

    texts      = [c["text"] for c in chunks]
    embeddings = []

    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        logger.info(f"Embedding batch {i // EMBED_BATCH + 1} ({len(batch)} chunks)...")
        embeddings.extend(embed_model.encode(batch).tolist())

    count = upsert_chunks(index, chunks, embeddings)
    logger.info(f"Synced {count} vectors to Pinecone.")
    return count


def run_pipeline():
    """
    Main pipeline entry point.
    Ingest → Chunk → Embed → Sync → Log
    """
    sf_logger = PipelineRunLogger()
    try:
        logger.info("Starting Knowledge Base Refresh Pipeline.")

        wiki_docs = fetch_wikipedia_articles()
        pdf_docs  = ingest_azure_pdfs()
        sf_logger.update(
            wiki_articles_fetched=len(wiki_docs),
            pdfs_ingested=len(pdf_docs),
        )

        all_chunks = chunk_documents(wiki_docs + pdf_docs)
        sf_logger.update(total_chunks_created=len(all_chunks))

        count = embed_and_sync(all_chunks)
        sf_logger.update(vectors_upserted=count)

        logger.info(f"Pipeline complete. {count} vectors processed.")
        sf_logger.commit(status="SUCCESS")

    except Exception as e:
        sf_logger.error_message = str(e)
        sf_logger.commit(status="FAILED")
        raise


if __name__ == "__main__":
    run_pipeline()
