import os
import time
import uuid
import logging
from typing import List, Tuple, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

import streamlit as st
import anthropic
from sentence_transformers import SentenceTransformer

from vector_store import get_pinecone_client, get_or_create_index, search
from snowflake_logger import log_chat_turn

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource
def initialize_services() -> Tuple[anthropic.Anthropic, SentenceTransformer, any]:
    """
    Initializes cloud clients and local ML models in a single cached call.
    Reads all credentials directly from environment variables / .env file.
    """
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    pinecone_key  = os.getenv("PINECONE_API_KEY")
    index_name    = os.getenv("PINECONE_INDEX_NAME")

    if not anthropic_key:
        st.error("ANTHROPIC_API_KEY is missing. Add it to your .env file or Streamlit secrets.")
        st.stop()
    if not pinecone_key:
        st.error("PINECONE_API_KEY is missing. Add it to your .env file or Streamlit secrets.")
        st.stop()
    if not index_name:
        st.error("PINECONE_INDEX_NAME is missing. Add it to your .env file or Streamlit secrets.")
        st.stop()

    try:
        ai_client  = anthropic.Anthropic(api_key=anthropic_key)
        pc_client  = get_pinecone_client(pinecone_key)
        index      = get_or_create_index(pc_client, index_name)
        model      = SentenceTransformer("all-MiniLM-L6-v2")
        return ai_client, model, index

    except Exception as e:
        logger.error(f"Initialization failure: {e}")
        st.error("Failed to connect to backend services. Check logs for details.")
        st.stop()


client, model, index = initialize_services()

st.set_page_config(
    page_title="Data Engineering Knowledge Assistant",
    page_icon="🧠",
    layout="wide"
)


def retrieve_context(question: str, n_results: int = 5) -> List[Dict[str, Any]]:
    """Performs semantic retrieval to find relevant knowledge chunks."""
    question_embedding = model.encode(question).tolist()
    return search(index, question_embedding, n_results)


def generate_rag_response(question: str):
    """
    Orchestrates the RAG flow: Retrieval -> Prompt Construction -> Streaming Generation.
    """
    chunks       = retrieve_context(question)
    context_text = ""
    for i, chunk in enumerate(chunks):
        context_text += f"\n--- Source {i+1}: {chunk['title']} ---\n{chunk['text']}\n"

    full_reply  = ""
    placeholder = st.empty()

    system_instruction = (
        "You are an expert Data Engineering and AI assistant. "
        "Answer the user's question using ONLY the context provided. "
        "If the answer is not in the context, state that you do not have "
        "enough information in your knowledge base. "
        "Maintain a professional, concise tone and use markdown for clarity."
    )

    try:
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system_instruction,
            messages=[{"role": "user", "content": f"CONTEXT:\n{context_text}\n\nQUESTION: {question}"}]
        ) as stream:
            for text in stream.text_stream:
                full_reply += text
                placeholder.markdown(full_reply + "▌")
    except Exception as e:
        logger.error(f"LLM Generation Error: {e}")
        full_reply = "I encountered an error generating a response. Please try again."

    placeholder.markdown(full_reply)
    return full_reply, chunks


# --- UI ---
st.title("🧠 Data Engineering Knowledge Assistant")
st.caption("Vector DB: Pinecone (Serverless) | Engine: Claude Sonnet | Orchestration: GitHub Actions")

with st.sidebar:
    st.header("Knowledge Base")
    st.markdown("""
    **Indexed Content**
    - 📖 Fundamentals of Data Engineering
    - 📖 The Data Engineer's Guide to Apache Spark
    - 📖 Generative AI & LLMs for Dummies
    - 📖 Cloud Data Engineering for Dummies
    - 📖 Cloud Data Warehousing for Dummies
    - 📖 Building Applications with Snowpark for Dummies
    - 🌐 Curated Wikipedia articles (ETL, MLOps, Snowflake, etc.)
    """)
    st.divider()

    try:
        stats = index.describe_index_stats()
        st.metric("Indexed Chunks", stats["total_vector_count"])
    except Exception:
        st.caption("Status: Knowledge base stats temporarily unavailable.")

    st.divider()
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("Reference Sources"):
                unique_sources = {chunk["title"]: chunk["url"] for chunk in message["sources"]}
                for title, url in unique_sources.items():
                    st.markdown(f"- [{title}]({url})")

if prompt := st.chat_input("Query the Data Engineering knowledge base..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant context..."):
            t0 = time.time()
            answer, chunks = generate_rag_response(prompt)
            elapsed_ms = int((time.time() - t0) * 1000)

            log_chat_turn(
                session_id       = st.session_state.session_id,
                question         = prompt,
                answer           = answer,
                chunks           = chunks,
                response_time_ms = elapsed_ms,
            )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": chunks
    })
    st.rerun()
