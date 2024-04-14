import os
import warnings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

warnings.simplefilter("ignore")

ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "db")


def create_vector_database():
    """
        Creates a vector database with embeddings generated from the document located in the ./data directory and
        persists into a Chromadb
    """
    pdf_loader = DirectoryLoader("./data/", glob="**/*.pdf", loader_cls=PyPDFLoader)
    loaded_documents = pdf_loader.load()

    # Splitting the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=40)
    chunked_documents = text_splitter.split_documents(loaded_documents)

    #Initialize the embeddings model
    ollama_embeddings = OllamaEmbeddings(model="mistral")

    # Create and persist a Chroma vector database from the chunked documents
    vector_database = Chroma.from_documents(
        documents=chunked_documents,
        embedding=ollama_embeddings,
        persist_directory=DB_DIR,
    )

    vector_database.persist()


if __name__ == '__main__':
    create_vector_database()
