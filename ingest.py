import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import CHROMA_PATH


def ingest():

    documents = []

    data_folder = "data"

    for file in os.listdir(data_folder):

        if file.endswith(".pdf"):

            loader = PyPDFLoader(os.path.join(data_folder, file))

            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    vectordb.persist()

    print("Vector database created successfully")


if __name__ == "__main__":
    ingest()