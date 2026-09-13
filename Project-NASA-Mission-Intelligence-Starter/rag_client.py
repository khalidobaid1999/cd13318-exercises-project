import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")
    
    chromadb_dirs = [d for d in current_dir.iterdir() if d.is_dir() and d.name.startswith("chroma_db")]
    for chromadb_dir in chromadb_dirs:
        try:
            client = chromadb.PersistentClient(path=str(chromadb_dir), settings=Settings(anonymized_telemetry=False))
            collections = client.list_collections()

            for collection in collections:
                key = f"{chromadb_dir.name}/{collection.name}"

                try:
                    count = client.get_collection(collection.name).count()
                except Exception:
                    count = "unknown"

                backends[key] = {
                    "path": str(chromadb_dir),
                    "collection_name": collection.name,
                    "display_name": f"{chromadb_dir.name} - {collection.name} ({count} documents)",
                    "count": count,
                }

        except Exception as e:
            backends[chromadb_dir.name] = {
                "path": str(chromadb_dir),
                "collection_name": None,
                "display_name": f"{chromadb_dir.name} (error: {str(e)[:50]})",
                "count": 0,
            }

    return backends

def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""
    client = chromadb.PersistentClient(path=chroma_dir, settings=Settings(anonymized_telemetry=False))
    return client.get_collection(collection_name)

def retrieve_documents(collection: chromadb.Collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    where_filter = None
    if mission_filter and mission_filter.lower() != "all":
        where_filter = {"mission": mission_filter}

    return collection.query(query_texts=[query],n_results=n_results, where=where_filter)


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""
    
    # TODO: Initialize list with header text for context section

    # TODO: Loop through paired documents and their metadata using enumeration
        # TODO: Extract mission information from metadata with fallback value
        # TODO: Clean up mission name formatting (replace underscores, capitalize)
        # TODO: Extract source information from metadata with fallback value  
        # TODO: Extract category information from metadata with fallback value
        # TODO: Clean up category name formatting (replace underscores, capitalize)
        
        # TODO: Create formatted source header with index number and extracted information
        # TODO: Add source header to context parts list
        
        # TODO: Check document length and truncate if necessary
        # TODO: Add truncated or full document content to context parts list

    # TODO: Join all context parts with newlines and return formatted string