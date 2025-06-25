# Disclaimer this file is largely untested and will need some work done to it first
# I'm considering removing any reference to nextcloud and making it a simple api that exposes some endpoints
# 1. An endpoint to ingest files to the vector db. 2. Another one to query the model through the llm_query and other files

import os
import logging
import asyncio
import time
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Thread, Event
from dataclasses import dataclass
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import uuid

from src.config import Config
from src.embeddings import DocumentEmbedder
from src.get_vector_db import get_vector_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize configuration
config = Config()

# Global variables
embedder = None
shutdown_event = Event()

# FastAPI app
app = FastAPI(
    title="RAG Webhook Service",
    description="Webhook service for handling Nextcloud file changes",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global embedder
    logger.info("Starting webhook service...")
    
    try:
        # Initialize embedder
        embedder = DocumentEmbedder(config)
        logger.info("Document embedder initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize webhook service: {str(e)}")
        raise


def is_supported_file(file_path: str) -> bool:
    """Check if file is supported for processing"""
    if not embedder:
        return False
    
    path = Path(file_path)
    return embedder.is_supported_file(path)

@app.post("/webhook/nextcloud")
async def nextcloud_webhook(payload: dict):
    """Handle NextCloud file change webhooks for specific files"""
    try:
        event_type = payload.get("event", {}).get("class", "")
        file_path = payload.get("event", {}).get("node", {}).get("path", "")
        user_id = payload.get("user", {}).get("uid", "")
        
        logger.info(f"NextCloud webhook: {event_type} for file {file_path} by user {user_id}")
        
        # Convert NextCloud path to actual filesystem path
        # NextCloud path: "/admin/files/Documents/report.pdf"
        # Filesystem path: "/mnt/ssd/nextcloud/data/admin/files/Documents/report.pdf"
        actual_file_path = convert_nextcloud_path_to_filesystem(file_path, user_id)
        
        if event_type == "OCP\\Files\\Events\\Node\\NodeCreatedEvent":
            # New file uploaded
            return await handle_file_created(actual_file_path)
            
        # elif event_type == "OCP\\Files\\Events\\Node\\NodeWrittenEvent":
        #     # File modified
        #     return await handle_file_modified(actual_file_path, background_tasks)
            
        # elif event_type == "OCP\\Files\\Events\\Node\\NodeDeletedEvent":
        #     # File deleted - remove from vector DB
        #     return await handle_file_deleted(actual_file_path, background_tasks)
            
        return {"status": "ignored", "reason": "Event type not handled"}
        
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}")
        return {"status": "error", "message": str(e)}

def convert_nextcloud_path_to_filesystem(nc_path: str, user_id: str) -> str:
    """Convert NextCloud internal path to actual filesystem path"""
    # Remove the user prefix and 'files' part
    # "/admin/files/Documents/report.pdf" -> "Documents/report.pdf"
    relative_path = nc_path.replace(f"/{user_id}/files/", "")
    
    # Construct full filesystem path
    # This depends on your NextCloudPi configuration
    base_path = f"{config.NEXTCLOUD_PATH}/{user_id}/files"  # Adjust for your setup
    return os.path.join(base_path, relative_path)

async def handle_file_created(file_path: str):
    """Process a single newly created file"""

    try:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
                raise Exception(f"File does not exist: {file_path}")
        
        if not embedder.is_supported_file(file_path_obj):
            return {"status": "failed","file": f"File type not supported: {file_path_obj.suffix}"}
        
            
        documents = embedder.load_document(file_path_obj)
        
        if not documents:
            raise Exception("Failed to load document")
        
        chunks = embedder.chunk_documents(documents)

        db = get_vector_db()
        db.add_documents(chunks)

        return {"status": "success","file": file_path}
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {str(e)}")
        return {"status": "failed","file": f"Failed to process file: {str(e)}"}