"""
=============================================================================
RAG Web Application Backend (FastAPI + ChromaDB)
=============================================================================
Hosts an interactive AI Web Interface that you and your colleagues can use
across your local Wi-Fi network or share via tunnel!
=============================================================================
"""

import os
import time
import shutil
import socket
import warnings
from typing import Optional

# Suppress benign warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_engine import RAGEngine

# Directory Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
DOCS_DIR = os.path.join(BASE_DIR, "sample_docs")
DB_DIR = os.path.join(BASE_DIR, "chroma_web_storage")

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Initialize RAG Engine
rag_engine = RAGEngine(db_path=DB_DIR, collection_name="web_knowledge_store")

# If collection is empty, automatically seed with sample documents
if rag_engine.collection.count() == 0:
    print("[Web RAG] Collection is empty. Auto-indexing initial sample docs...")
    rag_engine.ingest_directory(DOCS_DIR)

app = FastAPI(
    title="DSA RAG Tutor",
    description="Local, 100% Free RAG Engine with Modern Web UI",
    version="1.0.0"
)

# Enable CORS so colleagues on different devices/browsers can connect seamlessly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_local_ip() -> str:
    """Detects the machine's local Wi-Fi / LAN IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3
    relevance_threshold: Optional[float] = 0.65
    temperature: Optional[float] = 0.1


@app.get("/")
def serve_index():
    """Serves the main AI web interface."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_file):
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index_file)


@app.get("/api/share-info")
def get_share_info():
    """Returns local and network sharing URLs for colleagues."""
    ip = get_local_ip()
    port = 8000
    return {
        "local_url": f"http://localhost:{port}",
        "network_url": f"http://{ip}:{port}",
        "local_ip": ip,
        "port": port,
        "instructions": (
            f"Colleagues on the same Wi-Fi can open: http://{ip}:{port} on their phone, tablet, or laptop!"
        )
    }


@app.get("/api/stats")
def get_stats():
    """Returns vector database statistics and indexed files."""
    files = sorted([f for f in os.listdir(DOCS_DIR) if f.endswith((".pdf", ".txt", ".md"))])
    return {
        "total_chunks": rag_engine.collection.count(),
        "collection_name": rag_engine.collection_name,
        "indexed_documents": files,
        "embedding_model": "all-MiniLM-L6-v2 (ONNX Local)"
    }


@app.post("/api/sync")
def sync_docs_folder():
    """
    Scans the sample_docs directory and automatically indexes any new documents placed there.
    """
    existing_items = rag_engine.collection.get(include=["metadatas"])
    indexed_sources = set()
    if existing_items and "metadatas" in existing_items and existing_items["metadatas"]:
        for meta in existing_items["metadatas"]:
            if meta and "source" in meta:
                indexed_sources.add(meta["source"])

    files_on_disk = [f for f in os.listdir(DOCS_DIR) if f.endswith((".pdf", ".txt", ".md"))]
    newly_indexed = []
    chunks_added = 0

    for filename in sorted(files_on_disk):
        if filename not in indexed_sources:
            file_path = os.path.join(DOCS_DIR, filename)
            c_count = rag_engine.ingest_file(file_path)
            chunks_added += c_count
            newly_indexed.append(filename)

    return {
        "status": "success",
        "newly_indexed_files": newly_indexed,
        "new_chunks_added": chunks_added,
        "total_documents": len(files_on_disk),
        "total_chunks": rag_engine.collection.count(),
        "message": f"Sync complete! Indexed {len(newly_indexed)} new document(s) ({chunks_added} chunks added)."
    }


@app.post("/api/query")
def run_query(request: QueryRequest):
    """Executes the full RAG pipeline for a given user query."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    start_time = time.time()
    result = rag_engine.answer_query(
        query=request.query,
        top_k=request.top_k or 3,
        relevance_threshold=request.relevance_threshold or 0.65,
        temperature=request.temperature if request.temperature is not None else 0.1
    )
    elapsed_ms = round((time.time() - start_time) * 1000, 1)

    # Format chunks for clean JSON serialization
    serialized_chunks = []
    for c in result["retrieved_chunks"]:
        serialized_chunks.append({
            "content": c.content,
            "source": c.metadata.get("source", "Unknown"),
            "chunk_index": c.metadata.get("chunk_index", 0),
            "distance": c.distance,
            "relevance_score": c.relevance_score,
            "relevance_percent": round(c.relevance_score * 100, 1)
        })

    return {
        "query": request.query,
        "answer": result["answer"],
        "is_confident": result["is_confident"],
        "generator_used": result.get("generator_used", "Local Grounded Synthesizer"),
        "guardrail_status": result.get("guardrail_status"),
        "temperature": result.get("temperature", 0.1),
        "retrieved_chunks": serialized_chunks,
        "augmented_prompt": result.get("augmented_prompt"),
        "system_instruction": result.get("system_instruction"),
        "latency_ms": elapsed_ms
    }


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Receives and indexes a PDF, TXT, or Markdown document."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".pdf", ".txt", ".md"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Only .pdf, .txt, and .md are accepted."
        )

    safe_filename = os.path.basename(file.filename)
    saved_path = os.path.join(DOCS_DIR, safe_filename)
    try:
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Ingest into vector store
        chunks_added = rag_engine.ingest_file(saved_path)

        return {
            "filename": safe_filename,
            "status": "success",
            "chunks_added": chunks_added,
            "total_collection_chunks": rag_engine.collection.count(),
            "message": f"Successfully indexed '{safe_filename}' ({chunks_added} chunks added)."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@app.delete("/api/documents/{filename}")
def delete_document(filename: str):
    """
    Deletes the document from disk and purges all of its vector embeddings from ChromaDB.
    """
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(DOCS_DIR, safe_filename)

    # 1. Purge all chunks from ChromaDB vector database
    chunks_deleted = rag_engine.delete_document(safe_filename)

    # 2. Remove file from storage disk if present
    file_existed = False
    if os.path.exists(file_path):
        os.remove(file_path)
        file_existed = True

    if not file_existed and chunks_deleted == 0:
        raise HTTPException(status_code=404, detail=f"Document '{safe_filename}' not found.")

    return {
        "status": "success",
        "filename": safe_filename,
        "chunks_deleted": chunks_deleted,
        "total_collection_chunks": rag_engine.collection.count(),
        "message": f"Successfully deleted '{safe_filename}' and purged {chunks_deleted} vector chunk(s)."
    }


# Mount static assets
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    ip = get_local_ip()
    port = int(os.environ.get("PORT", 8000))
    print("=" * 65)
    print("🚀 DSA RAG Tutor Web Server Starting")
    print(f"👉 Local Access:   http://localhost:{port}")
    print(f"👉 Network Share:  http://{ip}:{port} (Share with colleagues on Wi-Fi!)")
    print("=" * 65)
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
