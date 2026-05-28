"""
utils/rag_pipeline.py
─────────────────────
Fixes:
  1. All imports use modern langchain_* packages — no deprecation warnings
  2. Content-hash deduplication — identical chunks never added twice
  3. MMR retriever — penalises redundant chunks at query time
  4. Chunk size 800 / overlap 80 — one complete idea per chunk
  5. Persistent FAISS index — survives Streamlit reruns without re-ingesting
"""

import os
import hashlib
import pickle
import shutil
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama3-8b-8192")
FAISS_DIR     = Path("faiss_store")
HASH_FILE     = FAISS_DIR / "doc_hashes.pkl"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 80

# ── Module-level singletons ───────────────────────────────────────────────────
_embeddings: HuggingFaceEmbeddings | None = None
_vectorstore: FAISS | None = None
_ingested_hashes: set = set()


# ── Internal helpers ──────────────────────────────────────────────────────────
def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
        )
    return _embeddings


def _get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
        length_function=len,
    )


def _get_llm() -> ChatGroq:
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
    )


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()


def _load_hashes() -> set:
    if HASH_FILE.exists():
        with open(HASH_FILE, "rb") as f:
            return pickle.load(f)
    return set()


def _save_hashes(hashes: set) -> None:
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    with open(HASH_FILE, "wb") as f:
        pickle.dump(hashes, f)


def _load_or_create_vectorstore() -> FAISS:
    global _ingested_hashes
    emb = _get_embeddings()

    if (FAISS_DIR / "index.faiss").exists():
        vs = FAISS.load_local(
            str(FAISS_DIR), emb, allow_dangerous_deserialization=True
        )
        _ingested_hashes = _load_hashes()
        return vs

    # Bootstrap with a seed doc so FAISS can initialise
    vs = FAISS.from_texts(["StudyPal knowledge base initialised."], emb)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(FAISS_DIR))
    _ingested_hashes = set()
    _save_hashes(_ingested_hashes)
    return vs


def _format_docs(docs):
    """Join retrieved doc contents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def _build_chain(vectorstore: FAISS):
    """
    Build RAG chain using langchain_core only — no langchain.chains imports.
    Returns a chain whose .invoke({"input": "..."}) returns:
        {"answer": "...", "context": [Document, ...]}
    """
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 20,
            "lambda_mult": 0.65,
        },
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are StudyPal, an expert study assistant. "
            "Answer the question using ONLY the context provided below. "
            "Be thorough but concise. Use bullet points or numbered lists where helpful. "
            "If the context does not contain enough information, say so clearly "
            "rather than making things up.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{input}"),
    ])

    llm = _get_llm()

    # Build chain manually using langchain_core primitives only
    def invoke_chain(inputs: dict) -> dict:
        question = inputs["input"]
        docs     = retriever.invoke(question)
        context  = _format_docs(docs)
        messages = prompt.format_messages(input=question, context=context)
        response = llm.invoke(messages)
        return {
            "answer":  response.content,
            "context": docs,
        }

    # Wrap as a simple callable that matches the .invoke({"input": ...}) API
    class RAGChain:
        def invoke(self, inputs: dict) -> dict:
            return invoke_chain(inputs)

    return RAGChain()


# ── Public API ────────────────────────────────────────────────────────────────
def load_rag():
    """Load (or initialise) the base RAG chain. Safe to call multiple times."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = _load_or_create_vectorstore()
    return _build_chain(_vectorstore)


def load_rag_with_docs(pdf_paths: list):
    """
    Ingest PDFs into the persistent FAISS store, skipping duplicates.
    Returns a fresh RAG chain backed by the updated store.
    """
    global _vectorstore, _ingested_hashes

    if _vectorstore is None:
        _vectorstore = _load_or_create_vectorstore()

    splitter   = _get_splitter()
    new_docs   = []
    skipped    = 0
    new_hashes = set()

    for path in pdf_paths:
        try:
            loader = PyPDFLoader(path)
            pages  = loader.load()
        except Exception as e:
            print(f"[RAG] Could not load {path}: {e}")
            continue

        chunks = splitter.split_documents(pages)

        for chunk in chunks:
            text = chunk.page_content.strip()
            if not text:
                continue

            h = _content_hash(text)

            if h in _ingested_hashes:
                skipped += 1
                continue

            chunk.metadata.setdefault("source", Path(path).name)
            chunk.metadata["content_hash"] = h
            new_docs.append(chunk)
            new_hashes.add(h)

    print(f"[RAG] New chunks: {len(new_docs)} | Skipped duplicates: {skipped}")

    if new_docs:
        _vectorstore.add_documents(new_docs)
        _vectorstore.save_local(str(FAISS_DIR))
        _ingested_hashes |= new_hashes
        _save_hashes(_ingested_hashes)

    return _build_chain(_vectorstore)


def reset_vectorstore():
    """Wipe the FAISS index and hash store completely."""
    global _vectorstore, _ingested_hashes
    if FAISS_DIR.exists():
        shutil.rmtree(FAISS_DIR)
    _vectorstore     = None
    _ingested_hashes = set()