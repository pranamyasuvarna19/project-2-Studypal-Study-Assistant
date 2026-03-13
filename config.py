import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Get API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_NAME = "llama-3.1-8b-instant"

CHROMA_PATH = "vectorstore"