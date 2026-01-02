"""Build unified vector index from all parsed documents.

Combines all chunks from parsed documents into a single searchable index.

Usage:
    python -m src.pipeline.build_unified_index
"""

import json
import pickle
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent.parent.parent
PARSED_DIR = PROJECT_ROOT / "data" / "out" / "parsed"
INDEX_DIR = PROJECT_ROOT / "data" / "out" / "unified_index"

# Embedding model
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024


def load_all_chunks() -> list[dict]:
    """Load chunks from all parsed documents."""
    all_chunks = []
    
    doc_dirs = sorted(PARSED_DIR.iterdir())
    print(f"Found {len(doc_dirs)} document directories")
    
    for doc_dir in doc_dirs:
        if not doc_dir.is_dir():
            continue
        
        chunks_file = doc_dir / "chunks.jsonl"
        if not chunks_file.exists():
            print(f"  ⚠️ No chunks.jsonl in {doc_dir.name}")
            continue
        
        doc_chunks = []
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunk = json.loads(line)
                    # Add source document info
                    chunk["source_doc"] = doc_dir.name
                    doc_chunks.append(chunk)
        
        print(f"  ✅ {doc_dir.name}: {len(doc_chunks)} chunks")
        all_chunks.extend(doc_chunks)
    
    return all_chunks


def build_bm25_index(chunks: list[dict]) -> tuple[BM25Okapi, list[str]]:
    """Build BM25 index from chunks."""
    texts = []
    tokenized = []
    
    for chunk in chunks:
        text = chunk.get("text", "")
        texts.append(text)
        # Simple whitespace tokenization for Finnish
        tokens = text.lower().split()
        tokenized.append(tokens)
    
    bm25 = BM25Okapi(tokenized)
    return bm25, texts


def build_vector_index(chunks: list[dict], model: SentenceTransformer) -> faiss.IndexFlatIP:
    """Build FAISS vector index from chunks."""
    texts = [chunk.get("text", "") for chunk in chunks]
    
    print(f"\nGenerating embeddings for {len(texts)} chunks...")
    start_time = time.time()
    
    # Generate embeddings in batches
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    
    embed_time = time.time() - start_time
    print(f"Embeddings generated in {embed_time:.1f}s ({len(texts)/embed_time:.1f} chunks/s)")
    
    # Create FAISS index
    embeddings_np = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings_np)
    
    return index


def save_index(
    chunks: list[dict],
    bm25: BM25Okapi,
    texts: list[str],
    faiss_index: faiss.IndexFlatIP,
    output_dir: Path,
) -> None:
    """Save all index artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save chunks metadata
    with open(output_dir / "chunks_metadata.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    # Save BM25
    with open(output_dir / "bm25.pkl", "wb") as f:
        pickle.dump((bm25, texts), f)
    
    # Save FAISS (use temp file to avoid encoding issues)
    import tempfile
    import shutil
    with tempfile.NamedTemporaryFile(delete=False, suffix=".index") as tmp:
        tmp_path = tmp.name
    faiss.write_index(faiss_index, tmp_path)
    shutil.move(tmp_path, str(output_dir / "faiss.index"))
    
    # Save metadata
    metadata = {
        "total_chunks": len(chunks),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "documents": list(set(c["source_doc"] for c in chunks)),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Index saved to: {output_dir}")


def main() -> None:
    print("=" * 60)
    print("BUILDING UNIFIED INDEX FOR ALL DOCUMENTS")
    print("=" * 60)
    
    # Load all chunks
    print("\n📄 Loading chunks from all documents...")
    chunks = load_all_chunks()
    print(f"\n📊 Total chunks: {len(chunks)}")
    
    if not chunks:
        print("❌ No chunks found!")
        sys.exit(1)
    
    # Build BM25
    print("\n🔍 Building BM25 index...")
    bm25, texts = build_bm25_index(chunks)
    print(f"BM25 vocabulary size: {len(bm25.idf)}")
    
    # Load embedding model
    print(f"\n🧠 Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL, device="cuda")
    
    # Build vector index
    faiss_index = build_vector_index(chunks, model)
    print(f"FAISS index size: {faiss_index.ntotal} vectors")
    
    # Save everything
    print("\n💾 Saving index...")
    save_index(chunks, bm25, texts, faiss_index, INDEX_DIR)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Documents:    {len(set(c['source_doc'] for c in chunks))}")
    print(f"Chunks:       {len(chunks)}")
    print(f"BM25 vocab:   {len(bm25.idf)}")
    print(f"FAISS vectors: {faiss_index.ntotal}")
    print(f"Index path:   {INDEX_DIR}")


if __name__ == "__main__":
    main()

