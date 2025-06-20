import streamlit as st
import logging
import time
import threading
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path
# Import your existing modules
from src.config import Config
from src.embeddings import DocumentEmbedder
from src.llm_query import query as rag_query
from src.get_vector_db import get_vector_db
import random
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize configuration
config = Config()

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'embedder' not in st.session_state:
    with st.spinner("Initializing RAG system..."):
        st.session_state.embedder = DocumentEmbedder(config)
        st.session_state.vector_db = get_vector_db()
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = None

for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "timestamp" in message:
                st.caption(f"_{message['timestamp']}_")

# Chat input
if prompt := st.chat_input("Ask me anything about your files..."):
    # Add user message to chat history
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.messages.append({
        "role": "user", 
        "content": prompt,
        "timestamp": timestamp
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(f"_{timestamp}_")

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching through your files..."):
            try:
                # Call your existing RAG query function
                response = rag_query(prompt)
                
                if response:
                    st.markdown(response)
                    response_timestamp = datetime.now().strftime("%H:%M:%S")
                    st.caption(f"_{response_timestamp}_")
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "timestamp": response_timestamp
                    })
                else:
                    error_msg = "I couldn't find relevant information to answer your question. Please try rephrasing or check if your files are properly indexed."
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                logger.error(f"Query error: {str(e)}")