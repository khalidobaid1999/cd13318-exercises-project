import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional, Tuple
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


def _overlap_length(first: str, second: str, max_overlap: int) -> int:
    min_overlap = max(1, max_overlap // 2)
    for k in range(min(max_overlap, len(first), len(second)), min_overlap - 1, -1):
        if first.endswith(second[:k]):
            return k
    return 0


def deduplicate_chunks(documents: List[str], metadatas: List[Dict]) -> List[Tuple[str, Dict]]:
    seen_texts = set()
    kept: List[Tuple[str, Dict]] = []

    for doc, meta in zip(documents, metadatas):
        meta = meta or {}
        text = (doc or "").strip()

        normalized = " ".join(text.split()).lower()
        if not normalized or normalized in seen_texts:
            continue

        doc_key = meta.get("file_path") or (meta.get("mission"), meta.get("source"))
        start, end = meta.get("chunk_start"), meta.get("chunk_end")

        if start is not None and end is not None:
            for kept_text, kept_meta in kept:
                kept_key = kept_meta.get("file_path") or (kept_meta.get("mission"), kept_meta.get("source"))
                kept_start, kept_end = kept_meta.get("chunk_start"), kept_meta.get("chunk_end")
                if kept_key != doc_key or kept_start is None or kept_end is None:
                    continue
                if start >= end or kept_start >= kept_end:
                    continue

                if kept_start <= start and end <= kept_end:
                    text = ""
                    break
                if kept_start < start < kept_end:
                    k = _overlap_length(kept_text, text, kept_end - start)
                    text = text[k:].lstrip()
                elif kept_start < end < kept_end:
                    k = _overlap_length(text, kept_text, end - kept_start)
                    text = text[:len(text) - k].rstrip()

        if not text:
            continue

        seen_texts.add(normalized)
        kept.append((text, meta))

    return kept


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""

    chunks = deduplicate_chunks(documents, metadatas)
    if not chunks:
        return ""

    context_parts = ["Here is relevant information from NASA mission documents:\n"]
    for i, (doc, meta) in enumerate(chunks, 1):
        mission = meta.get("mission", "Unknown Mission")
        mission = mission.replace("_", " ").title().capitalize()

        source = meta.get("source", "Unknown Source")

        category = meta.get("document_category", "Unknown Category")
        category = category.replace("_", " ").title().capitalize()

        context_parts.append(f"--- Source {i}: {mission} | {source} | {category} ---")

        max_chars = 1500
        if len(doc) > max_chars:
            context_parts.append(doc[:max_chars] + "... [truncated]")
        else:
            context_parts.append(doc)

        context_parts.append("")

    return "\n".join(context_parts)