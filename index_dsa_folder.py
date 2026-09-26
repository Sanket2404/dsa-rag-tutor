"""
=============================================================================
Scaler DSA Folder Batch Ingestion Utility
=============================================================================
Quickly index all 14 Scaler DSA Lecture PDFs into your local Vector Database!

Usage:
    python3 index_dsa_folder.py /path/to/your/scaler_dsa_folder

Example:
    python3 index_dsa_folder.py ~/Downloads/Scaler_DSA
=============================================================================
"""

import os
import sys
import glob
import shutil
from rag_engine import RAGEngine

def main():
    if len(sys.argv) < 2:
        print("=" * 65)
        print("📚 SCALER DSA BATCH INGESTION TOOL")
        print("=" * 65)
        print("Usage:")
        print("    python3 index_dsa_folder.py <path_to_dsa_folder>")
        print("\nExample:")
        print("    python3 index_dsa_folder.py ~/Downloads/Scaler_Lectures")
        print("=" * 65)
        return

    input_dir = os.path.expanduser(sys.argv[1])
    if not os.path.exists(input_dir):
        print(f"❌ Error: Directory not found: {input_dir}")
        return

    # Find all PDFs in the specified directory
    pdf_files = sorted(glob.glob(os.path.join(input_dir, "*.pdf")) + glob.glob(os.path.join(input_dir, "**/*.pdf"), recursive=True))

    if not pdf_files:
        print(f"⚠️ No PDF files found in: {input_dir}")
        return

    print("=" * 65)
    print(f"🚀 Found {len(pdf_files)} PDF lecture file(s) in: {input_dir}")
    print("=" * 65)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_docs_dir = os.path.join(base_dir, "sample_docs")
    db_dir = os.path.join(base_dir, "chroma_web_storage")
    os.makedirs(target_docs_dir, exist_ok=True)

    # Initialize RAG Engine
    engine = RAGEngine(db_path=db_dir, collection_name="web_knowledge_store")

    total_chunks = 0
    for idx, pdf_path in enumerate(pdf_files, 1):
        filename = os.path.basename(pdf_path)
        dest_path = os.path.join(target_docs_dir, filename)

        # Copy to local docs directory if not already there
        if os.path.abspath(pdf_path) != os.path.abspath(dest_path):
            shutil.copy2(pdf_path, dest_path)

        print(f"\n[{idx}/{len(pdf_files)}] Ingesting '{filename}'...")
        try:
            chunks_count = engine.ingest_file(dest_path)
            total_chunks += chunks_count
            print(f"   ✓ Chunks created: {chunks_count}")
        except Exception as e:
            print(f"   ❌ Failed to ingest {filename}: {e}")

    print("\n" + "=" * 65)
    print(f"🎉 All {len(pdf_files)} DSA lecture PDFs successfully indexed!")
    print(f"📊 Total vector chunks in ChromaDB: {engine.collection.count()}")
    print("👉 Now refresh http://localhost:8000 to start studying with AI!")
    print("=" * 65)

if __name__ == "__main__":
    main()
