from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from config import GROQ_API_KEY, MODEL_NAME, CHROMA_PATH


def load_rag():

    # Initialize LLM
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME,
        temperature=0.3
    )

    # Embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Load vector database
    vectordb = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    # Retriever
    retriever = vectordb.as_retriever(
        search_kwargs={"k": 3}
    )

    # Prompt template
    prompt = ChatPromptTemplate.from_template(
        """
You are an AI Study Assistant.

Use the provided study material to answer the student's question.

Context:
{context}

Question:
{input}

Instructions:
- Explain clearly like a teacher.
- Do NOT mention page numbers or metadata.
- Give simple explanations.
- If possible include examples.

Answer:
"""
    )

    # Create document chain
    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    # Create retrieval chain
    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )
    
    return retrieval_chain