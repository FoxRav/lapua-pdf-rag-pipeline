"""Query the unified index across all documents.

Usage:
    python -m src.pipeline.query_unified "Mikä on Lapuan strategia?"
    python -m src.pipeline.query_unified "Kuinka monta jäsentä valtuustossa on?"
"""

import argparse
import json
import pickle
import sys
import tempfile
import shutil
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent.parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "out" / "unified_index"

EMBEDDING_MODEL = "BAAI/bge-m3"


def load_index():
    """Load the unified index."""
    # Load chunks metadata
    with open(INDEX_DIR / "chunks_metadata.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    # Load BM25
    with open(INDEX_DIR / "bm25.pkl", "rb") as f:
        bm25, texts = pickle.load(f)
    
    # Load FAISS (copy to temp to avoid encoding issues)
    import tempfile
    import shutil
    faiss_path = INDEX_DIR / "faiss.index"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".index") as tmp:
        tmp_path = tmp.name
    shutil.copy(str(faiss_path), tmp_path)
    faiss_index = faiss.read_index(tmp_path)
    Path(tmp_path).unlink()  # Clean up temp file
    
    return chunks, bm25, texts, faiss_index


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
    
    # Normalize vector scores (already in 0-1 range for cosine similarity)
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
        chunk["vector_score"] = float(vector_scores[np.where(vector_indices == idx)[0][0]] if idx in vector_indices else 0)
        results.append(chunk)
    
    return results


def format_result(result: dict) -> str:
    """Format a single result for display."""
    lines = []
    lines.append(f"📄 {result.get('source_doc', 'unknown')} (sivu {result.get('page', '?')})")
    lines.append(f"   Score: {result['score']:.3f} (BM25: {result['bm25_score']:.3f}, Vector: {result['vector_score']:.3f})")
    
    text = result.get("text", "")[:300]
    if len(result.get("text", "")) > 300:
        text += "..."
    lines.append(f"   {text}")
    
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query unified index")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    args = parser.parse_args()
    
    print(f"🔍 Query: {args.query}")
    print("=" * 60)
    
    # Load index
    print("Loading index...")
    chunks, bm25, texts, faiss_index = load_index()
    print(f"Loaded {len(chunks)} chunks from {len(set(c['source_doc'] for c in chunks))} documents")
    
    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL, device="cuda")
    
    # Search
    print("Searching...")
    results = search_hybrid(args.query, chunks, bm25, texts, faiss_index, model, args.top_k)
    
    print("\n" + "=" * 60)
    print(f"TOP {len(results)} RESULTS")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] {format_result(result)}")


if __name__ == "__main__":
    main()

