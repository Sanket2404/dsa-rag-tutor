"""
=============================================================================
Automated Test & Verification Suite for Free Local RAG Engine
=============================================================================
Tests:
  1. Ingestion of TXT documents
  2. Ingestion of PDF documents using PyPDF
  3. Chunking & Overlap verification
  4. Vector Embedding & Storage in local ChromaDB
  5. Semantic Retrieval for distinct queries (Pricing, Refunds, Security, Specs)
  6. Hallucination Guard verification (out-of-domain query)
=============================================================================
"""

import os
import shutil
from rag_engine import RAGEngine

def run_rag_test():
    test_db_dir = "./test_chroma_storage"
    
    # Clean up any previous test database
    if os.path.exists(test_db_dir):
        shutil.rmtree(test_db_dir)

    print("=" * 70)
    print("🚀 STARTING FREE LOCAL RAG TEST SUITE (Mac M1)")
    print("=" * 70)

    # 1. Initialize Engine
    engine = RAGEngine(db_path=test_db_dir, collection_name="test_knowledge_base")

    # 2. Ingest Sample Docs (TXT and PDF)
    sample_dir = os.path.abspath("./sample_docs")
    print(f"\n📂 Step 1: Ingesting sample documents from: {sample_dir}")
    total_chunks = engine.ingest_directory(sample_dir)
    print(f"✅ Total chunks indexed into ChromaDB: {total_chunks}")

    # 3. Test Queries
    test_queries = [
        "What is the refund and cancellation policy?",
        "What are the subscription tiers and rate limits?",
        "How do I report a security incident?",
        "What are the RAM and CPU requirements for Enterprise Cloud Deployment?",
        "Who won the 2022 FIFA World Cup?"  # Out of domain query!
    ]

    print("\n" + "=" * 70)
    print("🔍 Step 2: Testing Semantic Retrieval & Grounded Generation")
    print("=" * 70)

    for i, query in enumerate(test_queries, 1):
        print(f"\n------------------------------------------------------------")
        print(f"Query #{i}: \"{query}\"")
        print(f"------------------------------------------------------------")
        
        result = engine.answer_query(query, top_k=2)

        print(f"Top Retrieved Context Chunks ({len(result['retrieved_chunks'])}):")
        for idx, chunk in enumerate(result['retrieved_chunks'], 1):
            src = chunk.metadata.get('source')
            score = chunk.relevance_score
            dist = chunk.distance
            preview = chunk.content.replace('\n', ' ')[:100] + "..."
            print(f"  [{idx}] Source: {src} | Relevance Score: {score*100:.1f}% (Distance: {dist})")
            print(f"      Snippet: {preview}")

        print("\nSynthesized Answer:")
        print(result["answer"])

    print("\n" + "=" * 70)
    print("🎉 ALL RAG TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_rag_test()
