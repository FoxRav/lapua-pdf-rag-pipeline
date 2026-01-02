"""Build hybrid search index (BM25 + vector + rerank).

Uses BGE-M3 via sentence-transformers for dense embeddings.
"""

import sys
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.common.io import get_output_dir, read_jsonl, write_json

# Embedding model choice - BGE-M3 supports Finnish and long documents
EMBEDDING_MODEL = "BAAI/bge-m3"


def load_embedding_model() -> SentenceTransformer:
    """Load BGE-M3 model with GPU support via sentence-transformers."""
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {EMBEDDING_MODEL} on {device}...")
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    return model


def create_bm25_index(chunks: list[dict[str, Any]]) -> tuple[BM25Okapi, list[str]]:
    """Create BM25 index from chunks."""
    texts = []
    for chunk in chunks:
        text = chunk.get("text", "")
        # Tokenize (simple whitespace split for Finnish)
        tokens = text.lower().split()
        texts.append(tokens)
    
    bm25 = BM25Okapi(texts)
    return bm25, [chunk.get("text", "") for chunk in chunks]


def create_vector_index(chunks: list[dict[str, Any]], model: SentenceTransformer) -> np.ndarray:
    """Create vector embeddings from chunks using BGE-M3."""
    texts = [chunk.get("text", "") for chunk in chunks]
    
    print(f"Encoding {len(texts)} texts with BGE-M3...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=8,  # BGE-M3 is larger, use smaller batch
        convert_to_numpy=True
    )
    return embeddings.astype("float32")


def save_faiss_index(embeddings: np.ndarray, index_path: Path) -> None:
    """Save FAISS index."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, str(index_path))


def hybrid_search(
    query: str,
    bm25: BM25Okapi,
    bm25_texts: list[str],
    vector_index: faiss.Index,
    vector_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    chunks: list[dict[str, Any]],
    bm25_weight: float = 0.4,
    vector_weight: float = 0.6,
    top_k: int = 50,
) -> list[tuple[int, float]]:
    """Perform hybrid search (BM25 + vector)."""
    # BM25 scores
    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
    
    # Vector scores (cosine similarity)
    distances, indices = vector_index.search(query_embedding.reshape(1, -1), top_k)
    vector_scores = np.zeros(len(chunks))
    for i, idx in enumerate(indices[0]):
        # Convert distance to similarity (L2 distance -> similarity)
        similarity = 1.0 / (1.0 + distances[0][i])
        vector_scores[idx] = similarity
    
    # Normalize vector scores
    if vector_scores.max() > 0:
        vector_scores = (vector_scores - vector_scores.min()) / (vector_scores.max() - vector_scores.min() + 1e-8)
    
    # Hybrid score
    hybrid_scores = bm25_weight * bm25_scores + vector_weight * vector_scores
    
    # Get top-k
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
    results = [(int(idx), float(hybrid_scores[idx])) for idx in top_indices]
    
    return results


def main(year: int) -> None:
    """Main index function."""
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    output_dir = get_output_dir(year)
    index_dir = output_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    
    # Load chunks
    section_chunks = read_jsonl(output_dir / "section_chunks.jsonl")
    table_chunks = read_jsonl(output_dir / "table_chunks.jsonl")
    statement_chunks = read_jsonl(output_dir / "statement_chunks.jsonl")
    
    all_chunks = section_chunks + table_chunks + statement_chunks
    
    if not all_chunks:
        print("No chunks found. Run chunk first.")
        return
    
    print(f"Indexing {len(all_chunks)} chunks...")
    
    # Load BGE-M3 embedding model
    model = load_embedding_model()
    
    # Create BM25 index
    print("Creating BM25 index...")
    bm25, bm25_texts = create_bm25_index(all_chunks)
    
    # Save BM25 (simple pickle for now)
    import pickle
    
    with (index_dir / "bm25.pkl").open("wb") as f:
        pickle.dump((bm25, bm25_texts), f)
    
    # Create vector index
    print("Creating vector embeddings...")
    embeddings = create_vector_index(all_chunks, model)
    
    # Save FAISS index
    print("Saving FAISS index...")
    save_faiss_index(embeddings, index_dir / "faiss.index")
    
    # Save metadata
    metadata = {
        "num_chunks": len(all_chunks),
        "embedding_dim": embeddings.shape[1],
        "model_name": EMBEDDING_MODEL,
        "chunk_ids": [chunk.get("chunk_id", f"chunk_{i}") for i, chunk in enumerate(all_chunks)],
    }
    write_json(metadata, index_dir / "metadata.json")
    
    # Save chunks metadata for retrieval (including text for RAG)
    chunks_metadata = []
    for i, chunk in enumerate(all_chunks):
        chunks_metadata.append(
            {
                "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
                "chunk_type": chunk.get("chunk_type", "unknown"),
                "year": chunk.get("year", year),
                "page": chunk.get("page", None),
                "section_path": chunk.get("section_path", ""),
                "table_id": chunk.get("table_id", None),
                "text": chunk.get("text", ""),  # Include text for RAG queries
            }
        )
    write_json(chunks_metadata, index_dir / "chunks_metadata.json")
    
    print(f"\nIndex complete. Saved to {index_dir}")
    print(f"  - BM25 index: {index_dir / 'bm25.pkl'}")
    print(f"  - FAISS index: {index_dir / 'faiss.index'}")
    print(f"  - Metadata: {index_dir / 'metadata.json'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline.04_index YEAR")
        sys.exit(1)
    
    year = int(sys.argv[1])
    main(year)

