"""
utils/graph_tab.py
──────────────────────────────────────────────────────────────────────────────
Renders the entire '🗺 Roadmap' tab in StudyPal.
Import and call render_graph_tab() inside `with tab_roadmap:` in app.py.

Sections:
  1. Index PDFs into the knowledge graph
  2. Ask a question using GraphRAG (local / global / hybrid / naive modes)
  3. Visualise the knowledge graph (interactive pyvis HTML)
  4. Learning path — ordered concept list extracted from the graph
"""

from __future__ import annotations

import os
import tempfile

import streamlit as st
import streamlit.components.v1 as components


# ── Pyvis graph renderer ──────────────────────────────────────────────────────
def _render_pyvis(nodes: list, edges: list, height: int = 600) -> str:
    """
    Build a Pyvis network from nodes/edges dicts and return the HTML string.
    Rendered inline via st.components.v1.html().
    """
    from pyvis.network import Network

    net = Network(
        height=f"{height}px",
        width="100%",
        bgcolor="#0d0f14",
        font_color="#e8eaf0",
        directed=False,
    )
    net.set_options("""
    {
      "nodes": {
        "font": { "size": 13, "face": "DM Sans" },
        "borderWidth": 1.5,
        "borderWidthSelected": 3
      },
      "edges": {
        "color": { "color": "#2a2f3e", "highlight": "#f0c040" },
        "font": { "size": 10, "color": "#7a8099" },
        "smooth": { "type": "dynamic" }
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 120,
          "springConstant": 0.08
        },
        "solver": "forceAtlas2Based",
        "stabilization": { "iterations": 150 }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 150,
        "hideEdgesOnDrag": true
      }
    }
    """)

    for node in nodes:
        net.add_node(
            node["id"],
            label=node["label"],
            title=node.get("title", node["label"]),
            size=node.get("size", 15),
            color=node.get("color", "#4a5568"),
        )

    for edge in edges:
        net.add_edge(
            edge["source"],
            edge["target"],
            title=edge.get("title", ""),
            label=edge.get("label", ""),
        )

    # Write to temp file and read back as HTML
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)
        tmp_path = f.name

    with open(tmp_path, "r", encoding="utf-8") as f:
        html = f.read()

    os.unlink(tmp_path)
    return html


# ── Main render function ──────────────────────────────────────────────────────
def render_graph_tab(subject: str) -> None:
    """Call this inside `with tab_roadmap:` in app.py."""

    # ── Import here so Streamlit hot-reload never sees a partially-bound name ─
    from utils.graph_rag_pipeline import (
        GRAPH_WORKING_DIR,
        get_graph_rag,
        get_graph_data,
        get_learning_path,
        insert_documents,
        query_graph,
    )

    # ── CSS for this tab ──────────────────────────────────────────────────────
    st.markdown("""
<style>
.mode-pill {
  display:inline-block;background:#1e2230;border:1px solid #2a2f3e;
  border-radius:100px;padding:4px 14px;font-size:.78rem;color:#7a8099;
  margin:3px;cursor:default
}
.mode-pill.active { background:rgba(240,192,64,.12);border-color:rgba(240,192,64,.3);color:#f0c040 }
.path-node {
  display:flex;align-items:center;gap:12px;background:#161922;
  border:1px solid #2a2f3e;border-radius:10px;padding:.75rem 1.1rem;margin-bottom:.5rem
}
.path-num { font-family:'Syne',sans-serif;font-weight:800;color:#f0c040;
            font-size:1.1rem;min-width:24px }
.path-label { font-size:.9rem;font-weight:500 }
</style>
""", unsafe_allow_html=True)

    # ── Info banner ───────────────────────────────────────────────────────────
    st.markdown("""
<div style="background:linear-gradient(135deg,#161922,#1a1e2e);border:1px solid rgba(240,192,64,.3);
            border-radius:14px;padding:1.4rem 1.6rem;margin-bottom:1.5rem">
  <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.1rem;margin-bottom:.4rem">
    🗺 GraphRAG Knowledge Graph
  </div>
  <div style="font-size:.85rem;color:#9aa0b4;line-height:1.7">
    Unlike standard RAG which retrieves isolated text chunks,
    <strong style="color:#f0c040">GraphRAG</strong> builds a knowledge graph of
    entities and relationships from your PDFs — enabling multi-hop reasoning,
    concept linking, and an interactive learning roadmap.<br>
    <span style="color:#7a8099;font-size:.8rem">
      Powered by LightRAG · Entity extraction via Groq llama-3.3-70b · Queries via llama-3.1-8b-instant
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Initialise LightRAG ───────────────────────────────────────────────────
    if "graph_rag" not in st.session_state:
        with st.spinner("Initialising GraphRAG engine…"):
            try:
                st.session_state.graph_rag = get_graph_rag()
            except Exception as e:
                st.error(f"Failed to initialise GraphRAG: {e}")
                return

    rag = st.session_state.graph_rag

    # ── Check if graph already exists ────────────────────────────────────────
    graphml_exists = os.path.exists(
        os.path.join(GRAPH_WORKING_DIR, "graph_chunk_entity_relation.graphml")
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Section 1 — Ingest PDFs into the knowledge graph
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div style="font-family:Syne,sans-serif;font-weight:700;font-size:1rem;'
        'margin-bottom:.6rem">📥 Index PDFs into Knowledge Graph</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:.85rem;color:#9aa0b4;margin-bottom:.8rem'>"
        "Upload PDFs to build/update the knowledge graph. "
        "LightRAG uses <strong style='color:#f0c040'>llama-3.3-70b</strong> "
        "to extract entities and relationships — allow 1–3 min per document."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── File uploader — always visible, not inside expander ──────────────────
    graph_pdfs = st.file_uploader(
        "Upload PDFs for GraphRAG",
        type=["pdf"],
        accept_multiple_files=True,
        key="graph_pdf_uploader",
        label_visibility="collapsed",
    )

    # Save uploaded file bytes to session state immediately so they survive
    # the Streamlit rerun that happens when the button is clicked.
    if graph_pdfs:
        st.session_state["_graph_pdf_bytes"] = [
            (f.name, f.read()) for f in graph_pdfs
        ]

    # Show the build button whenever we have files (uploaded now OR cached)
    cached_pdfs = st.session_state.get("_graph_pdf_bytes", [])
    if cached_pdfs:
        st.caption(f"📄 {len(cached_pdfs)} file(s) ready: {', '.join(n for n, _ in cached_pdfs)}")

        if st.button("🔨 Build Knowledge Graph", width="content"):
            # Write bytes to temp files — use delete=False + manual close
            # so Windows doesn't lock the file while PyPDFLoader reads it
            tmp_paths = []
            for fname, fbytes in cached_pdfs:
                fd, fpath = tempfile.mkstemp(suffix=".pdf")
                try:
                    os.write(fd, fbytes)
                finally:
                    os.close(fd)          # close fd before handing path to loader
                tmp_paths.append(fpath)

            progress = st.progress(0, text="Extracting text from PDFs…")
            try:
                with st.spinner(
                    f"Building knowledge graph from {len(tmp_paths)} PDF(s) "
                    f"— this may take a few minutes…"
                ):
                    char_count = insert_documents(rag, tmp_paths)
                progress.progress(100, text="Done!")
                st.success(
                    f"✅ Knowledge graph built from {len(tmp_paths)} PDF(s) "
                    f"({char_count:,} characters indexed). Scroll down to explore."
                )
                # Clear cached bytes after successful indexing
                st.session_state.pop("_graph_pdf_bytes", None)
                st.rerun()
            except Exception as e:
                st.error(f"Indexing error: {e}")
                import traceback
                st.code(traceback.format_exc(), language="python")
            finally:
                progress.empty()
                for p in tmp_paths:
                    try:
                        os.unlink(p)
                    except Exception:
                        pass

    if graphml_exists:
        nodes, edges = get_graph_data()
        st.success(
            f"✅ Knowledge graph active — **{len(nodes)} concepts**, **{len(edges)} relationships**"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Section 2 — Ask a question using GraphRAG
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown(
        '<div style="font-family:Syne,sans-serif;font-weight:700;font-size:1rem;'
        'margin-bottom:.8rem">💬 Ask via GraphRAG</div>',
        unsafe_allow_html=True,
    )

    if not graphml_exists:
        st.info("Index at least one PDF above to enable GraphRAG querying.")
    else:
        mode_descriptions = {
            "hybrid": "🔀 Hybrid — entity detail + community summaries (recommended)",
            "local":  "🔍 Local  — specific concept detail",
            "global": "🌐 Global — big-picture summary",
            "naive":  "📄 Naive  — plain vector search (no graph)",
        }

        col_q, col_mode = st.columns([3, 1])
        with col_q:
            graph_question = st.text_input(
                "Graph question",
                placeholder=f"e.g. How do the key concepts in {subject} relate to each other?",
                label_visibility="collapsed",
                key="graph_question_input",
            )
        with col_mode:
            query_mode = st.selectbox(
                "Mode",
                list(mode_descriptions.keys()),
                format_func=lambda m: mode_descriptions[m],
                label_visibility="collapsed",
                key="graph_mode_select",
            )

        if st.button("🔍 Query Knowledge Graph", width="content"):
            if graph_question.strip():
                with st.spinner(f"Traversing knowledge graph ({query_mode} mode)…"):
                    try:
                        answer = query_graph(rag, graph_question, mode=query_mode)
                        st.markdown(
                            f'<div class="answer-box">{answer}</div>',
                            unsafe_allow_html=True,
                        )
                        # Save to chat history
                        if "chat_history" in st.session_state:
                            st.session_state.chat_history.append(
                                (f"[GraphRAG/{query_mode}] {graph_question}", answer)
                            )
                    except Exception as e:
                        st.error(f"Query error: {e}")
            else:
                st.warning("Please enter a question.")

    # ══════════════════════════════════════════════════════════════════════════
    # Section 3 — Interactive Knowledge Graph Visualisation
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown(
        '<div style="font-family:Syne,sans-serif;font-weight:700;font-size:1rem;'
        'margin-bottom:.8rem">🕸 Interactive Knowledge Graph</div>',
        unsafe_allow_html=True,
    )

    if not graphml_exists:
        st.info("Index PDFs above to visualise the knowledge graph.")
    else:
        nodes, edges = get_graph_data()

        if not nodes:
            st.warning("Graph exists but has no nodes yet. Try re-indexing.")
        else:
            # Controls
            ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
            with ctrl_col1:
                max_nodes = st.slider("Max nodes to display", 20, 200, 80, key="graph_max_nodes")
            with ctrl_col2:
                graph_height = st.slider("Graph height (px)", 400, 900, 600, key="graph_height")
            with ctrl_col3:
                st.markdown(
                    "<br><div style='font-size:.78rem;color:#7a8099'>"
                    "🟡 Hub nodes &nbsp; 🟢 Connectors &nbsp; ⚫ Leaf nodes"
                    "</div>",
                    unsafe_allow_html=True,
                )

            # Trim to max_nodes (keep highest-degree nodes)
            display_nodes = sorted(nodes, key=lambda n: n["size"], reverse=True)[:max_nodes]
            display_ids   = {n["id"] for n in display_nodes}
            display_edges = [e for e in edges if e["source"] in display_ids and e["target"] in display_ids]

            # Node/edge stats
            st.caption(
                f"Showing {len(display_nodes)} of {len(nodes)} concepts · "
                f"{len(display_edges)} relationships visible"
            )

            with st.spinner("Rendering knowledge graph…"):
                graph_html = _render_pyvis(display_nodes, display_edges, height=graph_height)
                components.html(graph_html, height=graph_height + 20, scrolling=False)

            st.caption(
                "💡 Drag nodes to rearrange · Scroll to zoom · Hover for descriptions · "
                "Click a node to highlight its connections"
            )

    # ══════════════════════════════════════════════════════════════════════════
    # Section 4 — Learning Path / Roadmap
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown(
        '<div style="font-family:Syne,sans-serif;font-weight:700;font-size:1rem;'
        'margin-bottom:.8rem">🎯 Learning Path</div>',
        unsafe_allow_html=True,
    )

    if not graphml_exists:
        st.info("Index PDFs above to generate a learning path.")
    else:
        path_col1, path_col2 = st.columns([2, 1])
        with path_col1:
            start_topic = st.text_input(
                "Start from topic (leave blank for auto)",
                placeholder="e.g. Neural Network, Gradient Descent…",
                key="path_start_topic",
                label_visibility="visible",
            )
        with path_col2:
            path_length = st.slider("Path length", 5, 20, 10, key="path_length")

        if st.button("🗺 Generate Learning Path", width="content"):
            with st.spinner("Extracting learning path from knowledge graph…"):
                path = get_learning_path(
                    working_dir=GRAPH_WORKING_DIR,
                    start_topic=start_topic.strip(),
                    max_nodes=path_length,
                )

            if path:
                st.markdown(
                    "<div style='font-size:.82rem;color:#7a8099;margin-bottom:.8rem'>"
                    "Concepts ordered by connectivity — start from the top and work your way down."
                    "</div>",
                    unsafe_allow_html=True,
                )
                for i, concept in enumerate(path):
                    arrow = "→" if i < len(path) - 1 else "🎓"
                    st.markdown(
                        f'<div class="path-node">'
                        f'<div class="path-num">{i+1}</div>'
                        f'<div class="path-label">{concept}</div>'
                        f'<div style="margin-left:auto;color:#7a8099;font-size:.85rem">{arrow}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.warning("Could not extract a learning path. Try re-indexing your PDFs.")