"""
=============================================================================
Interactive RAG Terminal Explorer
=============================================================================
Run this script to interactively chat with your local documents!
Usage:
    python3 interactive_cli.py
=============================================================================
"""

import os
import sys
import warnings

# Suppress benign macOS LibreSSL / urllib3 warning for clean terminal outputs
warnings.filterwarnings("ignore")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from rag_engine import RAGEngine

console = Console()

def print_banner():
    banner_text = """
[bold cyan]🧠 DSA RAG TUTOR (Mac Apple Silicon Edition)[/bold cyan]
[dim]Powered by ChromaDB, Sentence-Transformers (all-MiniLM-L6-v2), & Apple Vision OCR[/dim]
[green]100% Offline • Zero API Costs • Private on Your Device[/green]
"""
    console.print(Panel(banner_text, border_style="cyan"))

def main():
    print_banner()
    
    # 1. Initialize Engine
    db_path = "./chroma_knowledge_store"
    engine = RAGEngine(db_path=db_path, collection_name="user_knowledge")
    
    # Check if collection is empty; if so, offer to ingest sample docs
    if engine.collection.count() == 0:
        sample_path = os.path.abspath("./sample_docs")
        console.print(f"[yellow]Your database is currently empty.[/yellow]")
        console.print(f"Auto-ingesting sample files from: [bold]{sample_path}[/bold]...")
        engine.ingest_directory(sample_path)
        console.print(f"[bold green]Indexed sample documents successfully![/bold green]\n")
    else:
        console.print(f"[green]✓ Connected to local Vector DB ({engine.collection.count()} chunks indexed).[/green]\n")

    while True:
        console.print("[bold yellow]Enter command or question:[/bold yellow]")
        console.print("[dim](Type 'add <filepath>' to index a new PDF/TXT file, 'stats' for DB info, or 'exit' to quit)[/dim]")
        
        try:
            user_input = input("\n👉 Query > ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Exiting RAG Explorer. Goodbye![/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "q"]:
            console.print("[cyan]Exiting RAG Explorer. Happy coding![/cyan]")
            break

        # Command: Add new document
        if user_input.lower().startswith("add "):
            file_to_add = user_input[4:].strip()
            if not os.path.exists(file_to_add):
                console.print(f"[red]Error: File not found: '{file_to_add}'[/red]")
                continue
            console.print(f"[cyan]Ingesting '{file_to_add}'...[/cyan]")
            try:
                chunks = engine.ingest_file(file_to_add)
                console.print(f"[green]✓ Successfully indexed {chunks} chunks from '{file_to_add}'![/green]\n")
            except Exception as e:
                console.print(f"[red]Failed to ingest file: {e}[/red]")
            continue

        # Command: Show DB stats
        if user_input.lower() == "stats":
            console.print(f"[cyan]Vector DB Path:[/cyan] {engine.db_path}")
            console.print(f"[cyan]Collection Name:[/cyan] {engine.collection_name}")
            console.print(f"[cyan]Total Indexed Chunks:[/cyan] {engine.collection.count()}\n")
            continue

        # Execute RAG Pipeline for query
        console.print(f"\n[bold magenta]Searching vector database for:[/bold magenta] \"{user_input}\"...")
        result = engine.answer_query(user_input, top_k=3, relevance_threshold=0.60)

        # Display Retrieved Chunks Table
        table = Table(title="🔍 Retrieved Context Chunks (Top-K Semantic Search)", border_style="blue")
        table.add_column("Rank", justify="center", style="bold cyan", width=6)
        table.add_column("Source File", style="yellow", width=22)
        table.add_column("Relevance", justify="right", style="green", width=12)
        table.add_column("Excerpt Preview", style="white")

        for idx, chunk in enumerate(result["retrieved_chunks"], 1):
            src = chunk.metadata.get("source", "Unknown")
            score = f"{chunk.relevance_score * 100:.1f}%"
            preview = chunk.content.replace("\n", " ")[:110] + "..."
            table.add_row(str(idx), src, score, preview)

        console.print(table)

        # Display Final Grounded Answer
        if result["is_confident"]:
            console.print(Panel(result["answer"], title="🤖 Grounded Answer (RAG Output)", border_style="green"))
        else:
            console.print(Panel(result["answer"], title="⚠️ Low Relevance Warning", border_style="yellow"))

        console.print("\n" + "─" * 60 + "\n")

if __name__ == "__main__":
    main()
