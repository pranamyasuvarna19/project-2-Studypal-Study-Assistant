"""
utils/graph_rag_pipeline.py  — rewritten for lightrag-hku v1.4+
──────────────────────────────────────────────────────────────────────────────
All async calls go through a single _run() helper so nest_asyncio handles
the Streamlit event loop conflict cleanly.

Key API facts:
  • rag.ainsert()   — async insert
  • rag.aquery()    — async query
  • rag.initialize_storages()   — MUST be awaited before any insert/query
  • initialize_pipeline_status() — MUST be called to prevent history errors
"""

from __future__ import annotations

import asyncio
import os
import threading
from typing import List, Tuple

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc, setup_logger
from lightrag.kg.shared_storage import initialize_pipeline_status

setup_logger("lightrag", level="WARNING")

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
GRAPH_WORKING_DIR = "./graph_rag_storage"
INDEXING_MODEL    = "llama-3.1-8b-instant"  # Kept 8b to prevent Groq Rate Limits
QUERY_MODEL       = "llama-3.1-8b-instant"
EMBED_MODEL       = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM         = 384
EMBED_MAX_TOKENS  = 512

# ── Thread & Event Loop Manager (Streamlit Fix) ───────────────────────────────
class _AsyncRunner:
    """
    Owns a single background thread with a persistent event loop.
    All LightRAG async calls are submitted here to avoid Streamlit thread crashes.
    """
    def __init__(self):
        self._loop   = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._start, daemon=True)
        self._thread.start()

    def _start(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=600)  # Extended timeout for heavy indexing


_runner: _AsyncRunner | None = None

def _get_runner() -> _AsyncRunner:
    global _runner
    if _runner is None:
        _runner = _AsyncRunner()
    return _runner

def _run(coro):
    """Submit a coroutine to the background event loop and block until done."""
    return _get_runner().run(coro)


# ── Embedding singleton ───────────────────────────────────────────────────────
_st_model = None

def _get_st_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(EMBED_MODEL)
    return _st_model

async def _hf_embed_func(texts: List[str]) -> np.ndarray:
    return _get_st_model().encode(texts, convert_to_numpy=True)


# ── Groq LLM functions with Rate Limit Backoff ────────────────────────────────
async def _groq_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list | None = None,
    model: str = INDEXING_MODEL,
    **kwargs,
) -> str:
    from groq import AsyncGroq, RateLimitError
    client = AsyncGroq(api_key=GROQ_API_KEY)
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    # Retry loop to handle 429 Too Many Requests safely
    for attempt in range(5):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=2000,
                temperature=0.1,
            )
            return resp.choices[0].message.content
        except RateLimitError:
            wait_time = 3 * (2 ** attempt)
            print(f"⚠️ Groq Rate Limit hit. Pausing {wait_time}s...")
            await asyncio.sleep(wait_time)
        except Exception as e:
            print(f"❌ Groq API Error: {e}")
            return ""
            
    return ""


async def _groq_index(prompt, system_prompt=None, history_messages=None, **kw) -> str:
    return await _groq_complete(prompt, system_prompt, history_messages,
                                model=INDEXING_MODEL, **kw)

async def _groq_query(prompt, system_prompt=None, history_messages=None, **kw) -> str:
    return await _groq_complete(prompt, system_prompt, history_messages,
                                model=QUERY_MODEL, **kw)


# ── LightRAG init ─────────────────────────────────────────────────────────────
async def _init_rag() -> LightRAG:
    os.makedirs(GRAPH_WORKING_DIR, exist_ok=True)
    rag = LightRAG(
        working_dir=GRAPH_WORKING_DIR,
        llm_model_func=_groq_index,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBED_DIM,
            max_token_size=EMBED_MAX_TOKENS,
            func=_hf_embed_func,
        ),
    )
    await rag.initialize_storages()      # required — or __aenter__ error
    await initialize_pipeline_status()   # required — or KeyError: history_messages
    return rag


# ── Singleton ─────────────────────────────────────────────────────────────────
_rag_instance: LightRAG | None = None

def get_graph_rag() -> LightRAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = _run(_init_rag())
    return _rag_instance


# ── Document ingestion ────────────────────────────────────────────────────────
def insert_documents(rag: LightRAG, pdf_paths: List[str]) -> int:
    from langchain_community.document_loaders import PyPDFLoader

    all_text = []
    for path in pdf_paths:
        pages = PyPDFLoader(path).load()
        text  = "\n\n".join(p.page_content for p in pages if p.page_content.strip())
        if text:
            all_text.append(text)

    if not all_text:
        raise ValueError("No text could be extracted from the uploaded PDFs.")

    combined = "\n\n".join(all_text)

    async def _do_insert():
        await rag.ainsert(combined)       # use async version explicitly

    _run(_do_insert())
    return len(combined)


# ── Querying ──────────────────────────────────────────────────────────────────
def query_graph(rag: LightRAG, question: str, mode: str = "hybrid") -> str:
    async def _do_query():
        return await rag.aquery(
            question,
            param=QueryParam(mode=mode),  # mode only — no other params
        )
    result = _run(_do_query())
    return result if isinstance(result, str) else str(result)


# ── Graph data for visualisation ──────────────────────────────────────────────
def get_graph_data(working_dir: str = GRAPH_WORKING_DIR) -> Tuple[list, list]:
    import networkx as nx

    path = os.path.join(working_dir, "graph_chunk_entity_relation.graphml")
    if not os.path.exists(path):
        return [], []

    G       = nx.read_graphml(path)
    degrees = dict(G.degree())
    max_deg = max(degrees.values(), default=1)

    def _color(deg):
        r = deg / max_deg
        if r > 0.66: return "#f0c040"
        if r > 0.33: return "#4ecca3"
        return "#4a5568"

    nodes = [
        {
            "id":    str(nid),
            "label": str(data.get("entity_name", data.get("id", nid)))[:40],
            "title": data.get("description", str(nid)),
            "size":  max(10, min(50, 10 + degrees.get(nid, 1) * 4)),
            "color": _color(degrees.get(nid, 1)),
        }
        for nid, data in G.nodes(data=True)
    ]
    edges = [
        {
            "source": str(s),
            "target": str(t),
            "label":  str(d.get("relationship", ""))[:30],
            "title":  d.get("description", ""),
        }
        for s, t, d in G.edges(data=True)
    ]
    return nodes, edges


# ── Learning path via BFS ─────────────────────────────────────────────────────
def get_learning_path(
    working_dir: str = GRAPH_WORKING_DIR,
    start_topic: str = "",
    max_nodes: int = 10,
) -> List[str]:
    import networkx as nx

    path = os.path.join(working_dir, "graph_chunk_entity_relation.graphml")
    if not os.path.exists(path):
        return []

    G    = nx.read_graphml(path)
    degs = dict(G.degree())
    if not degs:
        return []

    if start_topic:
        candidates = [
            n for n, d in G.nodes(data=True)
            if start_topic.lower() in str(d.get("entity_name", "")).lower()
        ]
        start = candidates[0] if candidates else max(degs, key=degs.get)
    else:
        start = max(degs, key=degs.get)

    visited, queue, seen = [], [start], set()
    while queue and len(visited) < max_nodes:
        node = queue.pop(0)
        if node in seen:
            continue
        seen.add(node)
        visited.append(G.nodes[node].get("entity_name", str(node)))
        queue.extend(
            sorted(G.neighbors(node), key=lambda n: degs.get(n, 0), reverse=True)
        )
    return visited