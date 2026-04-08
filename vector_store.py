import os
import time
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv(dotenv_path=Path(__file__).parent / ".env")


def get_pinecone_client(api_key: str = None) -> Pinecone:
    """
    Create a Pinecone client.
    Accepts an explicit api_key (preferred) or falls back to PINECONE_API_KEY env var.
    """
    key = api_key or os.getenv("PINECONE_API_KEY")
    if not key:
        raise ValueError("PINECONE_API_KEY is not set. Add it to your .env file or Streamlit secrets.")
    return Pinecone(api_key=key)


def get_or_create_index(pc: Pinecone, index_name: str):
    """Get existing Pinecone index or create a new one."""
    existing_indexes = [i.name for i in pc.list_indexes()]

    if index_name not in existing_indexes:
        print(f"Creating new Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print("Waiting for index to be ready...")
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
        print("Index ready.")
    else:
        print(f"Using existing index: {index_name}")

    return pc.Index(index_name)


def upsert_chunks(index, chunks: list, embeddings: list) -> int:
    """Store chunks and their embeddings in Pinecone in batches."""
    BATCH_SIZE = 100
    vectors = []

    for chunk, embedding in zip(chunks, embeddings):
        vectors.append({
            "id": chunk["id"],
            "values": embedding,
            "metadata": {
                "text":         chunk["text"],
                "title":        chunk["title"],
                "url":          chunk["url"],
                "chunk_index":  chunk["chunk_index"],
                "source_type":  chunk.get("source_type", "unknown"),
            }
        })

    for i in range(0, len(vectors), BATCH_SIZE):
        index.upsert(vectors=vectors[i:i + BATCH_SIZE])

    return len(vectors)


def search(index, query_embedding: list, n_results: int = 5) -> list:
    """Search Pinecone for the most similar chunks to a query embedding."""
    results = index.query(
        vector=query_embedding,
        top_k=n_results,
        include_metadata=True
    )

    return [
        {
            "text":  match["metadata"]["text"],
            "title": match["metadata"]["title"],
            "url":   match["metadata"]["url"],
            "score": match["score"],
        }
        for match in results["matches"]
    ]
