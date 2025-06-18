import os
import logging
from typing import Dict, List, Optional,Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from src.config import Config
from src.embeddings import DocumentEmbedder
from src.llm_query import query
from src.get_vector_db import get_vector_db

# Configure logging
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)

# Initialize configuration
config = Config()

# Global variables for application state
embedder = None
vector_db = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown."""
    global embedder, vector_db
    
    # Startup
    logger.info("Starting up RAG API server...")
    try:
        embedder = DocumentEmbedder(config)
        vector_db = get_vector_db()
        logger.info("Successfully initialized embedder and vector database")
    except Exception as e:
        logger.error(f"Failed to initialize components: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG API server...")

# Create FastAPI app with lifecycle management
app = FastAPI(
    title="RAG API Server",
    description="API for document ingestion and querying using RAG pipeline",
    version="1.0.0",
    lifespan=lifespan
)

# Pydantic models for request/response validation
class QueryRequest(BaseModel):
    query: str
    collection: Optional[str] = None

class QueryResponse(BaseModel):
    message: str
    sources: Optional[List[str]] = None

class IngestRequest(BaseModel):
    directory_path: Optional[str] = None
    collection: Optional[str] = None

class IngestResponse(BaseModel):
    message: str
    files_processed: int
    chunks_created: int

class StatusResponse(BaseModel):
    status: str
    ollama_model: str
    embedding_model: str
    collections_available: List[str]

class CollectionsResponse(BaseModel):
    collections: List[Dict[str, Any]]

# API Endpoints

@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """
    Ingest documents from the specified directory (or default) into the vector database.
    """
    try:
        logger.info(f"Starting document ingestion from: {request.directory_path or config.NEXTCLOUD_PATH}")
        
        # Get file stats before processing
        stats_before = embedder.get_file_stats(request.directory_path)
        
        # Process and embed documents
        success = embedder.embed_documents(request.directory_path)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to ingest documents")
        
        # Get updated stats (this is approximate since we don't track chunks per file yet)
        stats_after = embedder.get_file_stats(request.directory_path)
        
        logger.info("Document ingestion completed successfully")
        
        return IngestResponse(
            message="Documents ingested successfully",
            files_processed=stats_after.get('supported_files', 0),
            chunks_created=0  # TODO: Track this properly in embedder
        )
        
    except Exception as e:
        logger.error(f"Error during document ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the RAG system with a question and get an AI-generated response.
    """
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # Use the existing query function from llm_query.py
        response = query(request.query)
        
        if not response:
            raise HTTPException(status_code=500, detail="Failed to generate response")
        
        logger.info("Query processed successfully")
        
        return QueryResponse(
            message=response,
            sources=None  # TODO: Extract sources from retrieval context
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during query processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """
    Get the current status of the RAG system.
    """
    try:
        # TODO: Implement proper health checks
        # - Check if Ollama is running
        # - Check if vector DB is accessible
        # - Check if configured directories exist
        
        return StatusResponse(
            status="healthy",
            ollama_model=config.OLLAMA_MODEL,
            embedding_model=config.EMBEDDING_MODEL,
            collections_available=[config.DEFAULT_COLLECTION]  # Placeholder
        )
        
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get system status")

@app.get("/collections", response_model=CollectionsResponse)
async def list_collections():
    """
    List available document collections in the vector database.
    """
    try:
        # TODO: Implement proper collection listing
        # This depends on your vector DB implementation
        # For now, returning placeholder data
        
        collections = [
            {
                "name": config.DEFAULT_COLLECTION,
                "document_count": 0,  # TODO: Get actual count
                "last_updated": None,  # TODO: Track last update time
                "description": "Default document collection"
            }
        ]
        
        logger.info(f"Listed {len(collections)} collections")
        
        return CollectionsResponse(collections=collections)
        
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list collections")

# Additional utility endpoints

@app.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """
    Delete a specific collection from the vector database.
    """
    try:
        # TODO: Implement collection-specific deletion
        db = get_vector_db()
        
        if hasattr(db, 'delete_collection'):
            db.delete_collection()
            logger.info(f"Deleted collection: {collection_name}")
            return {"message": f"Collection '{collection_name}' deleted successfully"}
        else:
            raise HTTPException(status_code=501, detail="Collection deletion not implemented")
            
    except Exception as e:
        logger.error(f"Error deleting collection {collection_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete collection: {str(e)}")

@app.get("/files/stats")
async def get_file_stats(directory_path: Optional[str] = None):
    """
    Get statistics about files in the configured directory.
    """
    try:
        stats = embedder.get_file_stats(directory_path)
        return stats
    except Exception as e:
        logger.error(f"Error getting file stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get file statistics")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "message": "RAG API is running"}

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG API Server",
        "version": "1.0.0",
        "endpoints": {
            "POST /ingest": "Ingest documents into vector database",
            "POST /query": "Query documents using RAG",
            "GET /status": "Get system status",
            "GET /collections": "List available collections",
            "GET /health": "Health check"
        }
    }

# if __name__ == "__main__":
#     import uvicorn
    
#     host = os.getenv('API_HOST', '0.0.0.0')
#     port = int(os.getenv('API_PORT', '8000'))
#     debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
#     uvicorn.run(
#         "main:app", 
#         host=host, 
#         port=port, 
#         reload=debug,
#         log_level="info"
#     )