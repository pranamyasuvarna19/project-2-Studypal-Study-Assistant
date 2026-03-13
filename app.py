import streamlit as st

from utils.rag_pipeline import load_rag
from utils.video_search import get_video


# Load RAG pipeline
qa_chain = load_rag()

st.set_page_config(page_title="StudyPal", page_icon="📚")

st.title("📚 StudyPal - AI Study Assistant")


# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Sidebar
st.sidebar.header("Select Study Topic")

subject = st.sidebar.selectbox(
    "Subject",
    ["Python", "Machine Learning", "Data Structures"]
)

chapter = st.sidebar.text_input("Chapter / Topic")


# Question input
question = st.text_input("Ask your question")


if st.button("Get Explanation") and question:

    query = f"{subject} {chapter} {question}"

    result = qa_chain.invoke({"input": query})

    answer = result["answer"]

    # Save to history
    st.session_state.chat_history.append((question, answer))


    # Show answer
    st.subheader("📖 AI Explanation")
    st.write(answer)


    # Video references
    st.subheader("🎥 Video References")

    videos = get_video(query)

    for v in videos:
        st.markdown(f"[Watch Video]({v})")


# Display chat history
if st.session_state.chat_history:

    st.subheader("💬 Chat History")

    for q, a in reversed(st.session_state.chat_history):

        with st.expander(q):
            st.write(a)