import logging
from pathlib import Path
from typing import List, Optional
from unstructured.partition.auto import partition
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.get_vector_db import get_vector_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentEmbedder:
    """
    Handles document processing and embedding for RAG pipeline.
    Reads files from a specified directory and creates embeddings.
    """
    
    def __init__(self, config):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Smaller chunks for better retrieval
            chunk_overlap=200,  # More overlap for context preservation
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Supported file extensions by unstructured
        self.supported_extensions = {
            '.pdf', '.docx', '.doc', '.txt', '.md', '.html', '.htm',
            '.rtf', '.odt', '.csv', '.xlsx', '.xls', '.pptx', '.ppt'
        }
    
    def is_supported_file(self, file_path: Path) -> bool:
        """Check if file extension is supported by unstructured."""
        return file_path.suffix.lower() in self.supported_extensions
    
    def load_document(self, file_path: Path) -> List[Document]:
        """
        Load and process a single document using unstructured.
        Returns a list of LangChain Document objects.
        """
        try:
            # Use unstructured to partition the document
            elements = partition(filename=str(file_path))
            
            # Convert elements to text
            text_content = "\n\n".join([str(element) for element in elements])
            
            # Create LangChain Document with metadata
            document = Document(
                page_content=text_content,
                metadata={
                    "source": str(file_path),
                    "filename": file_path.name,
                    "file_type": file_path.suffix.lower(),
                    "file_size": file_path.stat().st_size,
                    "last_modified": file_path.stat().st_mtime
                }
            )
            
            return [document]
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            return []
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks for better retrieval."""
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
            return chunks
        except Exception as e:
            logger.error(f"Error chunking documents: {str(e)}")
            return []
    
    def process_directory(self, directory_path: Optional[str] = None) -> List[Document]:
        """
        Process all supported files in the specified directory.
        Returns a list of chunked documents ready for embedding.
        """
        if directory_path is None:
            directory_path = self.config.NEXTCLOUD_PATH
        
        dir_path = Path(directory_path)
        
        if not dir_path.exists():
            logger.error(f"Directory does not exist: {directory_path}")
            return []
        
        if not dir_path.is_dir():
            logger.error(f"Path is not a directory: {directory_path}")
            return []
        
        all_documents = []
        processed_files = 0
        
        # Recursively find all supported files
        for file_path in dir_path.rglob('*'):
            if file_path.is_file() and self.is_supported_file(file_path):
                logger.info(f"Processing file: {file_path.name}")
                
                documents = self.load_document(file_path)
                if documents:
                    all_documents.extend(documents)
                    processed_files += 1
        
        logger.info(f"Processed {processed_files} files, loaded {len(all_documents)} documents")
        
        # Chunk all documents
        if all_documents:
            chunked_documents = self.chunk_documents(all_documents)
            return chunked_documents
        
        return []
    
    def embed_documents(self, directory_path: Optional[str] = None) -> bool:
        """
        Main method to process directory and add documents to vector database.
        Returns True if successful, False otherwise.
        """
        try:
            # Process documents from directory
            chunks = self.process_directory(directory_path)
            
            if not chunks:
                logger.warning("No documents found to embed")
                return False
            
            # Get vector database and add documents
            db = get_vector_db()
            db.add_documents(chunks)
            
            # Persist the database (if using ChromaDB)
            if hasattr(db, 'persist'):
                db.persist()
            
            logger.info(f"Successfully embedded {len(chunks)} document chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error in embedding process: {str(e)}")
            return False
    
    def get_file_stats(self, directory_path: Optional[str] = None) -> dict:
        """
        Get statistics about files in the directory.
        Useful for monitoring and debugging.
        """
        if directory_path is None:
            directory_path = self.config.NEXTCLOUD_PATH
        
        dir_path = Path(directory_path)
        
        if not dir_path.exists():
            return {"error": "Directory does not exist"}
        
        stats = {
            "total_files": 0,
            "supported_files": 0,
            "file_types": {},
            "total_size": 0
        }
        
        for file_path in dir_path.rglob('*'):
            if file_path.is_file():
                stats["total_files"] += 1
                stats["total_size"] += file_path.stat().st_size
                
                if self.is_supported_file(file_path):
                    stats["supported_files"] += 1
                    ext = file_path.suffix.lower()
                    stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1
        
        return stats