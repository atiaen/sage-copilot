# backend/config.py
import os
from dataclasses import dataclass

@dataclass
class Config:
    # File paths
    NEXTCLOUD_PATH: str = os.getenv("NEXTCLOUD_PATH", "/home/deck/Documents/Books")
    
    # Model settings
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-minilm")
    
    # Collections
    DEFAULT_COLLECTION: str = os.getenv("COLLECTION_NAME", "documents")
    
    # ChromaDB
    CHROMA_PATH: str = os.getenv("CHROMA_PATH", "./chroma_db")