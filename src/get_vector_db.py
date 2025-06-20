import os
from src.config import Config
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores.chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

config= Config()

def get_vector_db():
    embedding = OllamaEmbeddings(model=config.EMBEDDING_MODEL)

    db = Chroma(
        collection_name=config.DEFAULT_COLLECTION,
        persist_directory=config.CHROMA_PATH,
        embedding_function=embedding
    )

    return db