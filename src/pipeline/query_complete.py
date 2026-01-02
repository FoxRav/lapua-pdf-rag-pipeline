"""Query the complete index (text + tables).

Usage:
    python -m src.pipeline.query_complete "Kuinka paljon on henkilöstöä?"
    python -m src.pipeline.query_complete "vakinaiset 470"
"""

import argparse
import json
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent.parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "out" / "complete_index"

EMBEDDING_MODEL = "BAAI/bge-m3"


def load_index():
    """Load the complete index."""
    # Load chunks metadata
    with open(INDEX_DIR / "chunks_metadata.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    # Load BM25
    with open(INDEX_DIR / "bm25.pkl", "rb") as f:
        bm25, texts = pickle.load(f)
    
    # Load FAISS (copy to temp for encoding issues)
    faiss_path = INDEX_DIR / "faiss.index"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".index") as tmp:
        tmp_path = tmp.name
    shutil.copy(str(faiss_path), tmp_path)
    faiss_index = faiss.read_index(tmp_path)
    Path(tmp_path).unlink()
    
    # Load metadata
    with open(INDEX_DIR / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    return chunks, bm25, texts, faiss_index, metadata


def search_hybrid(
    query: str,
    chunks: list,
    bm25,
    texts: list,
    faiss_index,
    model: SentenceTransformer,
    top_k: int = 10,
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
) -> list[dict]:
    """Perform hybrid search (BM25 + vector)."""
    # BM25 search
    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    
    # Normalize BM25 scores
    if bm25_scores.max() > 0:
        bm25_scores = bm25_scores / bm25_scores.max()
    
    # Vector search
    query_embedding = model.encode([query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding).astype("float32")
    vector_scores, vector_indices = faiss_index.search(query_embedding, len(chunks))
    
    vector_scores = vector_scores[0]
    vector_indices = vector_indices[0]
    
    # Combine scores
    combined_scores = np.zeros(len(chunks))
    combined_scores += bm25_weight * bm25_scores
    
    for i, idx in enumerate(vector_indices):
        combined_scores[idx] += vector_weight * vector_scores[i]
    
    # Get top results
    top_indices = np.argsort(combined_scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        chunk = chunks[idx].copy()
        chunk["score"] = float(combined_scores[idx])
        chunk["bm25_score"] = float(bm25_scores[idx])
        
        # Find vector score
        vec_idx = np.where(vector_indices == idx)[0]
        chunk["vector_score"] = float(vector_scores[vec_idx[0]]) if len(vec_idx) > 0 else 0.0
        
        results.append(chunk)
    
    return results


def format_result(result: dict, idx: int) -> str:
    """Format a single result for display."""
    lines = []
    
    chunk_type = result.get("chunk_type", result.get("element_type", "unknown"))
    source = result.get("source_doc", "unknown")
    page = result.get("page", "?")
    
    # Type indicator
    type_icon = "📊" if "table" in chunk_type else "📄"
    
    lines.append(f"[{idx}] {type_icon} {chunk_type.upper()} | {source} | Sivu {page}")
    lines.append(f"    Score: {result['score']:.3f} (BM25: {result['bm25_score']:.3f}, Vector: {result['vector_score']:.3f})")
    
    # Table evidence
    if "table_id" in result:
        lines.append(f"    Table: {result['table_id']}")
    
    # Text preview
    text = result.get("text", "")[:400]
    if len(result.get("text", "")) > 400:
        text += "..."
    lines.append(f"    {text}")
    
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query complete index")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--tables-only", action="store_true", help="Only search tables")
    parser.add_argument("--text-only", action="store_true", help="Only search text")
    args = parser.parse_args()
    
    print(f"🔍 Query: {args.query}")
    print("=" * 70)
    
    # Load index
    print("Loading index...")
    chunks, bm25, texts, faiss_index, metadata = load_index()
    
    print(f"Loaded {metadata['total_chunks']} chunks:")
    print(f"  - Text: {metadata['text_chunks']}")
    print(f"  - Tables: {metadata['table_chunks']}")
    print(f"  - Documents: {len(metadata['documents'])}")
    
    # Filter if needed
    if args.tables_only:
        filter_chunks = [c for c in chunks if c.get("chunk_type") == "table"]
        print(f"Filtering to {len(filter_chunks)} table chunks")
    elif args.text_only:
        filter_chunks = [c for c in chunks if c.get("chunk_type") == "text"]
        print(f"Filtering to {len(filter_chunks)} text chunks")
    else:
        filter_chunks = chunks
    
    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL, device="cuda")
    
    # Search
    print("Searching...")
    results = search_hybrid(args.query, chunks, bm25, texts, faiss_index, model, args.top_k)
    
    print("\n" + "=" * 70)
    print(f"TOP {len(results)} RESULTS")
    print("=" * 70)
    
    for i, result in enumerate(results, 1):
        print(f"\n{format_result(result, i)}")


if __name__ == "__main__":
    main()

