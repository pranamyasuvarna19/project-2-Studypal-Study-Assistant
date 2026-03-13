import os
import streamlit as st

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets["GROQ_API_KEY"]

MODEL_NAME = "llama-3.1-8b-instant"

CHROMA_PATH = "vectorstore"

