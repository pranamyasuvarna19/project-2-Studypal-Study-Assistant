import streamlit as st
import json
import os
import tempfile

# ── Load .env once here — all utils inherit the environment ──────────────────
from dotenv import load_dotenv
load_dotenv(override=False)   # override=False: never overwrite already-set env vars

from utils.rag_pipeline import load_rag, load_rag_with_docs
from utils.video_search import get_video
from utils.guardrails import run_guardrails
from utils.evaluation import run_evaluation
from utils.graph_tab import render_graph_tab
from utils.progress_tracker import init_db, save_quiz_result, save_question
from utils.progress_tab import render_progress_tab

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StudyPal",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Root Variables ── */
:root {
  --bg:         #0d0f14;
  --surface:    #161922;
  --surface2:   #1e2230;
  --border:     #2a2f3e;
  --accent:     #f0c040;
  --accent2:    #4ecca3;
  --accent3:    #e05c5c;
  --text:       #e8eaf0;
  --muted:      #7a8099;
  --radius:     14px;
}

/* ── Global Reset ── */
html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  font-family: 'DM Sans', sans-serif;
  color: var(--text);
}
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Typography ── */
h1, h2, h3, .syne { font-family: 'Syne', sans-serif !important; }

/* ── Hero Header ── */
.hero {
  background: linear-gradient(135deg, #161922 0%, #1a1e2e 60%, #0f1520 100%);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 2.5rem 3rem;
  margin-bottom: 2rem;
  position: relative;
  overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute; inset: 0;
  background: radial-gradient(ellipse 60% 50% at 80% 50%, rgba(240,192,64,.08) 0%, transparent 70%);
  pointer-events: none;
}
.hero-tag {
  display: inline-block;
  background: rgba(240,192,64,.12);
  color: var(--accent);
  border: 1px solid rgba(240,192,64,.3);
  border-radius: 100px;
  padding: 4px 14px;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: .08em;
  text-transform: uppercase;
  margin-bottom: .8rem;
}
.hero h1 {
  font-size: 2.6rem;
  font-weight: 800;
  margin: 0 0 .5rem;
  line-height: 1.1;
}
.hero h1 span { color: var(--accent); }
.hero p {
  color: var(--muted);
  font-size: 1rem;
  margin: 0;
  font-weight: 300;
}

/* ── Cards ── */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.6rem;
  margin-bottom: 1.2rem;
  transition: border-color .2s;
}
.card:hover { border-color: rgba(240,192,64,.35); }
.card-title {
  font-family: 'Syne', sans-serif;
  font-size: 1rem;
  font-weight: 700;
  letter-spacing: .03em;
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 8px;
}
.card-title .icon {
  width: 30px; height: 30px;
  background: rgba(240,192,64,.12);
  border-radius: 8px;
  display: inline-flex; align-items: center; justify-content: center;
  font-size: .9rem;
}

/* ── Answer Box ── */
.answer-box {
  background: var(--surface2);
  border-left: 3px solid var(--accent);
  border-radius: 0 var(--radius) var(--radius) 0;
  padding: 1.2rem 1.4rem;
  font-size: .95rem;
  line-height: 1.7;
  color: var(--text);
}

/* ── Quiz Card ── */
.quiz-card {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.4rem;
  margin-bottom: 1rem;
}
.quiz-q {
  font-family: 'Syne', sans-serif;
  font-size: .95rem;
  font-weight: 600;
  margin-bottom: .8rem;
}
.score-display {
  background: linear-gradient(135deg, rgba(78,204,163,.12), rgba(240,192,64,.08));
  border: 1px solid var(--accent2);
  border-radius: var(--radius);
  padding: 1.4rem;
  text-align: center;
  font-family: 'Syne', sans-serif;
}
.score-big {
  font-size: 3rem;
  font-weight: 800;
  color: var(--accent);
}

/* ── Voice Button ── */
.voice-area {
  background: var(--surface2);
  border: 1px dashed var(--border);
  border-radius: var(--radius);
  padding: 1.2rem;
  text-align: center;
  margin-bottom: 1rem;
}
.voice-hint { font-size: .8rem; color: var(--muted); margin-top: .4rem; }

/* ── Upload Zone ── */
.upload-label {
  font-size: .78rem;
  font-weight: 600;
  letter-spacing: .06em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: .5rem;
}

/* ── Video Pills ── */
.video-pill {
  display: inline-block;
  background: rgba(78,204,163,.1);
  border: 1px solid rgba(78,204,163,.25);
  color: var(--accent2) !important;
  border-radius: 100px;
  padding: 5px 14px;
  font-size: .82rem;
  text-decoration: none;
  margin: 4px 4px 4px 0;
}
.video-pill:hover { background: rgba(78,204,163,.2); }

/* ── Streamlit Widget Overrides ── */
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px rgba(240,192,64,.15) !important;
}
div[data-testid="stSelectbox"] > div > div {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
}
div.stButton > button {
  background: var(--accent) !important;
  color: #0d0f14 !important;
  border: none !important;
  border-radius: 10px !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  font-size: .88rem !important;
  letter-spacing: .03em !important;
  padding: .55rem 1.4rem !important;
  transition: opacity .15s, transform .1s !important;
}
div.stButton > button:hover {
  opacity: .88 !important;
  transform: translateY(-1px) !important;
}
div.stButton > button:active { transform: translateY(0) !important; }

/* Secondary button variant via label trick */
div.stButton > button[kind="secondary"] {
  background: var(--surface2) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
}

[data-testid="stFileUploader"] {
  background: var(--surface2) !important;
  border: 1px dashed var(--border) !important;
  border-radius: var(--radius) !important;
  padding: .8rem !important;
}
[data-testid="stFileUploader"] * { color: var(--text) !important; }

div[data-testid="stRadio"] label { color: var(--text) !important; }
div[data-testid="stRadio"] div[role="radio"] { border-color: var(--accent) !important; }

.stSuccess, [data-testid="stAlert"][data-baseweb="notification"] {
  background: rgba(78,204,163,.08) !important;
  border: 1px solid rgba(78,204,163,.3) !important;
  border-radius: 10px !important;
}
.stError {
  background: rgba(224,92,92,.08) !important;
  border: 1px solid rgba(224,92,92,.3) !important;
  border-radius: 10px !important;
}
.stWarning {
  background: rgba(240,192,64,.08) !important;
  border: 1px solid rgba(240,192,64,.3) !important;
  border-radius: 10px !important;
}
div[data-testid="stExpander"] {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
}
hr { border-color: var(--border) !important; }

/* Tab styling */
button[data-baseweb="tab"] {
  font-family: 'Syne', sans-serif !important;
  font-weight: 600 !important;
  color: var(--muted) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
  color: var(--accent) !important;
  border-bottom-color: var(--accent) !important;
}
[data-testid="stTabContent"] {
  background: transparent !important;
  padding-top: 1.2rem !important;
}

/* Sidebar label */
[data-testid="stSidebar"] .sidebar-logo {
  font-family: 'Syne', sans-serif;
  font-size: 1.3rem;
  font-weight: 800;
  padding: 1rem 0 1.5rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1.5rem;
  color: var(--accent);
}
[data-testid="stSidebar"] .sidebar-section {
  font-size: .72rem;
  font-weight: 700;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 1.4rem 0 .6rem;
}

/* Progress bar */
[data-testid="stProgressBar"] > div { background: var(--surface2) !important; }
[data-testid="stProgressBar"] > div > div { background: var(--accent) !important; }

/* Spinner */
[data-testid="stSpinner"] * { color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

# ─── Voice-to-Text JS Component ───────────────────────────────────────────────
voice_component = """
<div class="voice-area" id="voiceArea">
  <button onclick="toggleRecording()" id="voiceBtn"
    style="background:#f0c040;border:none;border-radius:10px;
           padding:10px 24px;font-family:'Syne',sans-serif;
           font-weight:700;font-size:.88rem;cursor:pointer;
           color:#0d0f14;transition:all .2s;">
    🎙 Start Voice Input
  </button>
  <div class="voice-hint" id="voiceHint">Click to speak your question</div>
  <div id="transcript" style="margin-top:.8rem;font-size:.9rem;color:#e8eaf0;
       min-height:1.5rem;font-style:italic;"></div>
</div>
<script>
let recognition = null;
let isRecording = false;

function toggleRecording() {
  if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
    document.getElementById('voiceHint').textContent = '⚠ Speech recognition not supported in this browser.';
    return;
  }
  if (isRecording) {
    recognition.stop();
    return;
  }
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.lang = 'en-US';

  recognition.onstart = () => {
    isRecording = true;
    document.getElementById('voiceBtn').textContent = '⏹ Stop Recording';
    document.getElementById('voiceBtn').style.background = '#e05c5c';
    document.getElementById('voiceHint').textContent = '🔴 Listening…';
  };
  recognition.onresult = (e) => {
    let interim = '', final = '';
    for (let i = e.resultIndex; i < e.results.length; i++) {
      if (e.results[i].isFinal) final += e.results[i][0].transcript;
      else interim += e.results[i][0].transcript;
    }
    document.getElementById('transcript').textContent = final || interim;
    if (final) {
      // Send to Streamlit via query param trick
      const inputs = window.parent.document.querySelectorAll('input[aria-label="Ask your question"]');
      if (inputs.length) {
        const nativeInput = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value');
        nativeInput.set.call(inputs[0], final);
        inputs[0].dispatchEvent(new Event('input', {bubbles:true}));
      }
    }
  };
  recognition.onend = () => {
    isRecording = false;
    document.getElementById('voiceBtn').textContent = '🎙 Start Voice Input';
    document.getElementById('voiceBtn').style.background = '#f0c040';
    document.getElementById('voiceHint').textContent = 'Click to speak your question';
  };
  recognition.start();
}
</script>
"""

# ─── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_resource
def get_qa_chain(doc_hash=None):
    """Cache the QA chain; rebuild only when new docs are added."""
    return load_rag()


def get_qa_chain_with_docs(tmp_paths):
    return load_rag_with_docs(tmp_paths)


def generate_quiz(qa_chain, topic: str, n: int = 5):
    prompt = f"""Generate {n} multiple choice questions about "{topic}".
Return ONLY a valid JSON array — no markdown, no extra text:
[
  {{
    "question": "...",
    "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
    "answer": "A",
    "explanation": "Brief explanation of the correct answer."
  }}
]"""
    result = qa_chain.invoke({"input": prompt})
    raw = result.get("answer", "")
    # Strip markdown fences if present
    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        return json.loads(raw)
    except Exception:
        st.error("Quiz generation failed — the model returned unexpected output. Try again.")
        return []


# ─── Session State Init ────────────────────────────────────────────────────────
defaults = {
    "chat_history": [],
    "quiz": None,
    "score": 0,
    "answered": [],
    "uploaded_docs": [],
    "qa_chain": None,
    "quiz_n": 3,
    "eval_history": [],   # list of EvalResult objects
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── DB Init (runs once, idempotent) ──────────────────────────────────────────
init_db()

# ─── QA chain bootstrap ────────────────────────────────────────────────────────
if st.session_state.qa_chain is None:
    with st.spinner("Loading AI engine…"):
        st.session_state.qa_chain = get_qa_chain()

qa_chain = st.session_state.qa_chain

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">📚 StudyPal</div>', unsafe_allow_html=True)

    # ── Subject ──
    st.markdown('<div class="sidebar-section">Study Topic</div>', unsafe_allow_html=True)
    subject = st.selectbox(
        "Subject",
        ["Python", "Machine Learning", "Data Structures", "Mathematics",
         "Statistics", "Deep Learning", "Computer Networks", "Other"],
        label_visibility="collapsed",
    )

    # ── PDF Upload ──
    st.markdown('<div class="sidebar-section">Upload Documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Upload lecture notes or textbooks to ground the AI answers.",
    )

    # Cache file bytes immediately — survives the rerun caused by button click
    if uploaded_files:
        st.session_state["_sidebar_pdf_bytes"] = [
            (f.name, f.read()) for f in uploaded_files
        ]

    cached_sidebar_pdfs = st.session_state.get("_sidebar_pdf_bytes", [])

    if cached_sidebar_pdfs:
        if st.button("📥 Process PDFs"):
            with st.spinner("Ingesting documents…"):
                tmp_paths = []
                # mkstemp + os.close() avoids Windows file-lock issue
                for fname, fbytes in cached_sidebar_pdfs:
                    fd, fpath = tempfile.mkstemp(suffix=".pdf")
                    try:
                        os.write(fd, fbytes)
                    finally:
                        os.close(fd)
                    tmp_paths.append(fpath)
                try:
                    st.session_state.qa_chain = get_qa_chain_with_docs(tmp_paths)
                    qa_chain = st.session_state.qa_chain
                    st.session_state.uploaded_docs = [n for n, _ in cached_sidebar_pdfs]
                    st.session_state.pop("_sidebar_pdf_bytes", None)
                    st.success(f"✓ {len(tmp_paths)} PDF(s) ingested!")
                except Exception as e:
                    st.error(f"Error processing PDFs: {e}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")
                finally:
                    for p in tmp_paths:
                        try:
                            os.unlink(p)
                        except Exception:
                            pass

    if st.session_state.uploaded_docs:
        st.markdown(
            "**Active documents:**<br>" +
            "".join(f"• {d}<br>" for d in st.session_state.uploaded_docs),
            unsafe_allow_html=True,
        )

    # ── Quiz settings ──
    st.markdown('<div class="sidebar-section">Quiz Settings</div>', unsafe_allow_html=True)
    st.session_state.quiz_n = st.slider("Number of questions", 3, 10, st.session_state.quiz_n)

    # ── Clear history ──
    st.markdown("---")
    if st.button("🗑 Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# ─── Main Content ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-tag">AI Study Assistant</div>
  <h1>Learn Smarter with <span>StudyPal</span></h1>
  <p>Ask questions, generate quizzes, and explore video references — all powered by your documents.</p>
</div>
""", unsafe_allow_html=True)

tab_ask, tab_quiz, tab_history, tab_eval, tab_roadmap, tab_progress = st.tabs([
    "💬 Ask & Explain", "🧠 Quiz Mode", "📜 History", "📊 Evaluate", "🗺 Roadmap", "📈 Progress"
])

# ═══════════════════════════════════════════════════════════
# TAB 1 — Ask & Explain
# ═══════════════════════════════════════════════════════════
with tab_ask:
    col_q, col_hint = st.columns([3, 1])

    with col_q:
        st.markdown('<div class="card-title"><span class="icon">✏️</span>Your Question</div>',
                    unsafe_allow_html=True)

        # Voice input component
        st.components.v1.html(voice_component, height=140)

        question = st.text_input(
            "Ask your question",
            placeholder=f"e.g. Explain recursion in {subject} in simple terms…",
            label_visibility="visible",
        )

    with col_hint:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.info("💡 **Tip:** Upload PDFs in the sidebar so answers are grounded in your notes.")

    btn_col1, btn_col2, _ = st.columns([1, 1, 3])
    with btn_col1:
        explain_clicked = st.button("🔍 Get Explanation", width="stretch")
    with btn_col2:
        if st.button("🎲 Surprise Me", width="stretch"):
            question = f"Give me an interesting fact or tip about {subject}"

    if explain_clicked and question:
        # ── Guardrails ────────────────────────────────────────────────────────
        guard = run_guardrails(question)
        if not guard.allowed:
            st.warning(guard.reason)
        else:
            if guard.pii_found:
                st.info(guard.pii_summary)

            query = f"{subject} {guard.clean_text}".strip()
            with st.spinner("Thinking…"):
                try:
                    result = qa_chain.invoke({"input": query})
                    answer = result.get("answer", "No answer returned.")
                    # Store original question in history (not the scrubbed version)
                    st.session_state.chat_history.append((question, answer))
                    # Track question in progress DB
                    save_question(subject, question)

                    # Save context chunks for evaluation tab
                    source_docs = result.get("context", [])
                    context_texts = [doc.page_content for doc in source_docs]
                    st.session_state.eval_history.append({
                        "question": guard.clean_text,
                        "answer": answer,
                        "contexts": context_texts,
                        "subject": subject,
                    })

                    st.markdown('<div class="card-title"><span class="icon">📖</span>AI Explanation</div>',
                                unsafe_allow_html=True)
                    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

                    if source_docs:
                        with st.expander("📄 Source Chunks"):
                            for i, doc in enumerate(source_docs[:3]):
                                st.markdown(f"**Chunk {i+1}:** {doc.page_content[:300]}…")

                    # Videos
                    st.markdown("---")
                    st.markdown('<div class="card-title"><span class="icon">🎥</span>Video References</div>',
                                unsafe_allow_html=True)
                    videos = get_video(query)
                    if videos:
                        links = "".join(
                            f'<a class="video-pill" href="{v}" target="_blank">▶ Watch {i+1}</a>'
                            for i, v in enumerate(videos)
                        )
                        st.markdown(links, unsafe_allow_html=True)
                    else:
                        st.caption("No videos found for this query.")
                except Exception as e:
                    st.error(f"Error: {e}")
    elif explain_clicked:
        st.warning("Please type a question first.")

# ═══════════════════════════════════════════════════════════
# TAB 2 — Quiz Mode
# ═══════════════════════════════════════════════════════════
with tab_quiz:
    gen_col, _ = st.columns([1, 3])
    with gen_col:
        if st.button(f"🧠 Generate {st.session_state.quiz_n}-Question Quiz", width="stretch"):
            topic = subject
            with st.spinner(f"Creating quiz on {topic}…"):
                quiz = generate_quiz(qa_chain, topic, st.session_state.quiz_n)
                if quiz:
                    st.session_state.quiz = quiz
                    st.session_state.score = 0
                    st.session_state.answered = [False] * len(quiz)

    if st.session_state.quiz:
        quiz = st.session_state.quiz
        answered_count = sum(st.session_state.answered)

        # Progress
        st.progress(answered_count / len(quiz))
        st.caption(f"Progress: {answered_count}/{len(quiz)} questions answered")
        st.markdown("---")

        for i, q in enumerate(quiz):
            with st.container():
                st.markdown(
                    f'<div class="quiz-card"><div class="quiz-q">Q{i+1}. {q["question"]}</div></div>',
                    unsafe_allow_html=True,
                )
                user_ans = st.radio(
                    f"q{i+1}",
                    q["options"],
                    key=f"q_{i}",
                    label_visibility="collapsed",
                )
                check_col, _ = st.columns([1, 4])
                with check_col:
                    if st.button(f"Check ✓", key=f"check_{i}", width="stretch",
                                 disabled=st.session_state.answered[i]):
                        correct_letter = q["answer"].strip().upper()
                        selected_letter = user_ans.split(".")[0].strip().upper()
                        if selected_letter == correct_letter:
                            st.success("✅ Correct!")
                            if not st.session_state.answered[i]:
                                st.session_state.score += 1
                        else:
                            correct_option = next(
                                (opt for opt in q["options"] if opt.strip().upper().startswith(correct_letter)),
                                correct_letter,
                            )
                            st.error(f"❌ Incorrect — Correct: **{correct_option}**")

                        if "explanation" in q:
                            st.info(f"💡 {q['explanation']}")

                        st.session_state.answered[i] = True

        # Final score
        if all(st.session_state.answered):
            pct = st.session_state.score / len(quiz)
            grade = "🏆 Excellent!" if pct >= .8 else ("👍 Good job!" if pct >= .5 else "📖 Keep studying!")

            # ── Save to progress DB (only once, guard with session flag) ──────
            save_key = f"_quiz_saved_{id(quiz)}"
            if not st.session_state.get(save_key):
                save_quiz_result(subject, st.session_state.score, len(quiz))
                st.session_state[save_key] = True
            st.markdown(f"""
<div class="score-display">
  <div style="font-size:.85rem;color:#7a8099;font-family:'Syne',sans-serif;
              letter-spacing:.08em;text-transform:uppercase;margin-bottom:.4rem;">
    Final Score
  </div>
  <div class="score-big">{st.session_state.score}<span style="font-size:1.5rem;color:#7a8099;">/{len(quiz)}</span></div>
  <div style="font-size:1.1rem;margin-top:.4rem;">{grade}</div>
</div>
""", unsafe_allow_html=True)

            st.markdown("&nbsp;")
            reset_col, _ = st.columns([1, 3])
            with reset_col:
                if st.button("🔄 New Quiz", width="stretch"):
                    st.session_state.quiz = None
                    st.session_state.score = 0
                    st.session_state.answered = []
                    st.rerun()

# ═══════════════════════════════════════════════════════════
# TAB 3 — Chat History
# ═══════════════════════════════════════════════════════════
with tab_history:
    if not st.session_state.chat_history:
        st.markdown("""
<div style="text-align:center;padding:3rem;color:#7a8099;">
  <div style="font-size:2.5rem;margin-bottom:.8rem;">💬</div>
  <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:600;">No history yet</div>
  <div style="font-size:.9rem;margin-top:.3rem;">Your explanations will appear here.</div>
</div>""", unsafe_allow_html=True)
    else:
        st.caption(f"{len(st.session_state.chat_history)} conversation(s) recorded")
        for idx, (q, a) in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"💬 {q[:80]}{'…' if len(q) > 80 else ''}"):
                st.markdown(f'<div class="answer-box">{a}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# TAB 4 — DeepEval Evaluation
# ═══════════════════════════════════════════════════════════
with tab_eval:
    st.markdown('<div class="card-title"><span class="icon">📊</span>RAG Evaluation with DeepEval</div>',
                unsafe_allow_html=True)

    # ── Info banner ───────────────────────────────────────────────────────────
    st.markdown("""
<div class="card" style="border-color:rgba(240,192,64,.3);">
  <div style="font-size:.9rem;line-height:1.7;color:#b0b8cc;">
    Evaluate your RAG pipeline using <strong style="color:#f0c040;">5 DeepEval metrics</strong>:
    <span style="color:#4ecca3;">Answer Relevancy</span> ·
    <span style="color:#4ecca3;">Faithfulness</span> ·
    <span style="color:#4ecca3;">Contextual Relevancy</span> ·
    <span style="color:#7a8099;">Contextual Precision*</span> ·
    <span style="color:#7a8099;">Contextual Recall*</span><br>
    <span style="font-size:.8rem;color:#7a8099;">* Requires an expected/reference answer to be provided.</span>
  </div>
</div>
""", unsafe_allow_html=True)

    if not st.session_state.eval_history:
        st.markdown("""
<div style="text-align:center;padding:3rem;color:#7a8099;">
  <div style="font-size:2.5rem;margin-bottom:.8rem;">📊</div>
  <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:600;">No interactions to evaluate yet</div>
  <div style="font-size:.9rem;margin-top:.3rem;">Ask a question in the <strong>Ask & Explain</strong> tab first, then come back here.</div>
</div>""", unsafe_allow_html=True)
    else:
        # ── Select which interaction to evaluate ──────────────────────────────
        eval_options = {
            f"Q{len(st.session_state.eval_history) - i}: {e['question'][:70]}…"
            if len(e['question']) > 70
            else f"Q{len(st.session_state.eval_history) - i}: {e['question']}": i
            for i, e in enumerate(reversed(st.session_state.eval_history))
        }
        selected_label = st.selectbox(
            "Select an interaction to evaluate",
            list(eval_options.keys()),
            label_visibility="visible",
        )
        selected_idx = len(st.session_state.eval_history) - 1 - eval_options[selected_label]
        selected = st.session_state.eval_history[selected_idx]

        # ── Show the interaction details ──────────────────────────────────────
        with st.expander("📋 View Interaction Details", expanded=False):
            st.markdown(f"**Question:** {selected['question']}")
            st.markdown(f'<div class="answer-box">{selected["answer"]}</div>', unsafe_allow_html=True)
            if selected["contexts"]:
                st.markdown(f"**Retrieved {len(selected['contexts'])} context chunk(s)**")
                for i, ctx in enumerate(selected["contexts"][:3]):
                    st.caption(f"Chunk {i+1}: {ctx[:200]}…")

        # ── Optional reference answer ─────────────────────────────────────────
        expected = st.text_area(
            "Reference answer (optional — enables Contextual Precision & Recall)",
            placeholder="Paste the expected/ideal answer here to unlock all 5 metrics…",
            height=80,
        )

        # ── Run evaluation ────────────────────────────────────────────────────
        eval_col, _ = st.columns([1, 3])
        with eval_col:
            run_eval = st.button("▶ Run Evaluation", width="stretch")

        if run_eval:
            if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GROQ_API_KEY"):
                st.error(
                    "⚠️ No judge LLM key found. Add one of these to your `.env` file:\n\n"
                    "```\nGOOGLE_API_KEY=AIza...   # recommended — Gemini\n"
                    "GROQ_API_KEY=gsk_...      # fallback — already set for the RAG chain\n```"
                )
            else:
                with st.spinner("Running DeepEval metrics — this takes 15–30 seconds…"):
                    try:
                        eval_result = run_evaluation(
                            question=selected["question"],
                            answer=selected["answer"],
                            contexts=selected["contexts"],
                            expected=expected.strip(),
                        )

                        st.markdown("---")
                        st.markdown(
                            f'<div class="card-title"><span class="icon">🎯</span>'
                            f'Results — {"✅ All Passed" if eval_result.overall_pass else "⚠️ Some Failed"}'
                            f'<span style="font-size:.75rem;color:#7a8099;font-family:DM Sans,sans-serif;'
                            f'font-weight:400;margin-left:10px;">judge: {eval_result.judge_model}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        # ── Metric score cards ────────────────────────────────
                        metrics = list(eval_result.scores.keys())
                        cols = st.columns(max(len(metrics), 1))

                        for col, name in zip(cols, metrics):
                            score   = eval_result.scores[name]
                            passed  = eval_result.passed[name]
                            reason  = eval_result.reasons.get(name, "")
                            error   = eval_result.errors.get(name, "")
                            color   = "#4ecca3" if passed else "#e05c5c"
                            icon    = "✅" if passed else "❌"
                            pct     = int(score * 100)

                            with col:
                                st.markdown(f"""
<div class="quiz-card" style="text-align:center;border-color:{color}40;">
  <div style="font-size:.72rem;font-weight:700;letter-spacing:.07em;
              text-transform:uppercase;color:#7a8099;margin-bottom:.5rem;">
    {name}
  </div>
  <div style="font-size:2rem;font-weight:800;color:{color};font-family:'Syne',sans-serif;">
    {pct}<span style="font-size:1rem;">%</span>
  </div>
  <div style="font-size:.85rem;margin-top:.3rem;">{icon} {"Pass" if passed else "Fail"}</div>
</div>""", unsafe_allow_html=True)
                                if error:
                                    st.caption(f"⚠ {error[:120]}")
                                elif reason:
                                    with st.expander("Why?"):
                                        st.write(reason)

                        # ── Skipped metrics note ──────────────────────────────
                        if eval_result.skipped_metrics:
                            st.caption(
                                f"⏭ Skipped (no reference answer provided): "
                                + ", ".join(eval_result.skipped_metrics)
                            )

                        # ── Overall badge ─────────────────────────────────────
                        st.markdown("---")
                        overall_color = "#4ecca3" if eval_result.overall_pass else "#e05c5c"
                        overall_label = "All metrics passed ✅" if eval_result.overall_pass else "One or more metrics failed ⚠️"
                        st.markdown(
                            f'<div style="text-align:center;padding:.8rem;background:rgba({("78,204,163" if eval_result.overall_pass else "224,92,92")},.1);'
                            f'border:1px solid {overall_color}40;border-radius:12px;'
                            f'font-family:Syne,sans-serif;font-weight:700;color:{overall_color};">'
                            f'{overall_label}</div>',
                            unsafe_allow_html=True,
                        )

                    except Exception as e:
                        st.error(f"Evaluation error: {e}")

        # ── Clear eval history ────────────────────────────────────────────────
        st.markdown("---")
        if st.button("🗑 Clear Evaluation History"):
            st.session_state.eval_history = []
            st.rerun()

# ═══════════════════════════════════════════════════════════
# TAB 5 — GraphRAG Roadmap
# ═══════════════════════════════════════════════════════════
with tab_roadmap:
    render_graph_tab(subject)

# ═══════════════════════════════════════════════════════════
# TAB 6 — Student Progress
# ═══════════════════════════════════════════════════════════
with tab_progress:
    render_progress_tab()