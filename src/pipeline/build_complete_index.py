"""Build complete index: 25 PDF text + Lapua 2024 tables.

Combines:
1. All 25 PDF text chunks (from batch_ingest)
2. Lapua 2024 table chunks (from PaddleOCR)

Usage:
    python -m src.pipeline.build_complete_index
"""

import json
import pickle
import shutil
import sys
import tempfile
import time
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent.parent.parent
PARSED_DIR = PROJECT_ROOT / "data" / "out" / "parsed"
LAPUA_2024_DIR = PROJECT_ROOT / "data" / "out" / "2024"
INDEX_DIR = PROJECT_ROOT / "data" / "out" / "complete_index"

# Embedding model
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# Version info
PARSER_VERSION = "2.0.0"
CHUNKER_VERSION = "2.0.0"
INDEX_VERSION = "2.0.0"


def load_text_chunks_from_batch() -> list[dict]:
    """Load text chunks from all 25 parsed PDFs."""
    all_chunks = []
    
    doc_dirs = sorted(PARSED_DIR.iterdir())
    print(f"\n📄 Loading text chunks from {len(list(PARSED_DIR.iterdir()))} documents...")
    
    for doc_dir in doc_dirs:
        if not doc_dir.is_dir():
            continue
        
        chunks_file = doc_dir / "chunks.jsonl"
        if not chunks_file.exists():
            continue
        
        doc_chunks = []
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunk = json.loads(line)
                    chunk["source_doc"] = doc_dir.name
                    chunk["chunk_type"] = "text"
                    doc_chunks.append(chunk)
        
        if doc_chunks:
            print(f"  ✅ {doc_dir.name}: {len(doc_chunks)} text chunks")
        all_chunks.extend(doc_chunks)
    
    return all_chunks


def load_table_chunks_lapua_2024() -> list[dict]:
    """Load table chunks from Lapua 2024 (PaddleOCR)."""
    tables_path = LAPUA_2024_DIR / "tables.jsonl"
    
    if not tables_path.exists():
        print(f"  ⚠️ No tables.jsonl found at {tables_path}")
        return []
    
    chunks = []
    with open(tables_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunk = json.loads(line)
                chunk["source_doc"] = "lapua_tilinpaatos_2024"
                chunk["chunk_type"] = "table"
                chunks.append(chunk)
    
    print(f"  ✅ Lapua 2024 tables: {len(chunks)} table chunks")
    return chunks


def build_bm25_index(chunks: list[dict]) -> tuple[BM25Okapi, list[str]]:
    """Build BM25 index from chunks."""
    texts = []
    tokenized = []
    
    for chunk in chunks:
        text = chunk.get("text", "")
        texts.append(text)
        tokens = text.lower().split()
        tokenized.append(tokens)
    
    bm25 = BM25Okapi(tokenized)
    return bm25, texts


def build_vector_index(chunks: list[dict], model: SentenceTransformer) -> faiss.IndexFlatIP:
    """Build FAISS vector index from chunks."""
    texts = [chunk.get("text", "") for chunk in chunks]
    
    print(f"\n🧮 Generating embeddings for {len(texts)} chunks...")
    start_time = time.time()
    
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    
    embed_time = time.time() - start_time
    print(f"⏱️ Embeddings generated in {embed_time:.1f}s ({len(texts)/embed_time:.1f} chunks/s)")
    
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
    """Save all index artifacts with versioning."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save chunks metadata
    with open(output_dir / "chunks_metadata.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    # Save BM25
    with open(output_dir / "bm25.pkl", "wb") as f:
        pickle.dump((bm25, texts), f)
    
    # Save FAISS (use temp file for encoding issues)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".index") as tmp:
        tmp_path = tmp.name
    faiss.write_index(faiss_index, tmp_path)
    shutil.move(tmp_path, str(output_dir / "faiss.index"))
    
    # Save metadata with versioning
    text_chunks = [c for c in chunks if c.get("chunk_type") == "text"]
    table_chunks = [c for c in chunks if c.get("chunk_type") == "table"]
    
    metadata = {
        "total_chunks": len(chunks),
        "text_chunks": len(text_chunks),
        "table_chunks": len(table_chunks),
        "documents": list(set(c.get("source_doc", "") for c in chunks)),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "parser_version": PARSER_VERSION,
        "chunker_version": CHUNKER_VERSION,
        "index_version": INDEX_VERSION,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Index saved to: {output_dir}")


def main() -> None:
    print("=" * 70)
    print("BUILDING COMPLETE INDEX (25 PDF text + Lapua 2024 tables)")
    print("=" * 70)
    
    # Load all chunks
    text_chunks = load_text_chunks_from_batch()
    table_chunks = load_table_chunks_lapua_2024()
    
    all_chunks = text_chunks + table_chunks
    
    print(f"\n📊 TOTAL: {len(all_chunks)} chunks")
    print(f"   - Text chunks: {len(text_chunks)}")
    print(f"   - Table chunks: {len(table_chunks)}")
    
    if not all_chunks:
        print("❌ No chunks found!")
        sys.exit(1)
    
    # Build BM25
    print("\n🔍 Building BM25 index...")
    bm25, texts = build_bm25_index(all_chunks)
    print(f"   BM25 vocabulary: {len(bm25.idf)} terms")
    
    # Load embedding model
    print(f"\n🧠 Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL, device="cuda")
    
    # Build vector index
    faiss_index = build_vector_index(all_chunks, model)
    print(f"   FAISS vectors: {faiss_index.ntotal}")
    
    # Save everything
    print("\n💾 Saving index...")
    save_index(all_chunks, bm25, texts, faiss_index, INDEX_DIR)
    
    # Summary
    print("\n" + "=" * 70)
    print("COMPLETE INDEX SUMMARY")
    print("=" * 70)
    print(f"Documents:     {len(set(c.get('source_doc', '') for c in all_chunks))}")
    print(f"Text chunks:   {len(text_chunks)}")
    print(f"Table chunks:  {len(table_chunks)}")
    print(f"Total chunks:  {len(all_chunks)}")
    print(f"BM25 vocab:    {len(bm25.idf)}")
    print(f"FAISS vectors: {faiss_index.ntotal}")
    print(f"Index path:    {INDEX_DIR}")
    print(f"\nVersions:")
    print(f"  parser:  {PARSER_VERSION}")
    print(f"  chunker: {CHUNKER_VERSION}")
    print(f"  index:   {INDEX_VERSION}")
    print(f"  embed:   {EMBEDDING_MODEL}")


if __name__ == "__main__":
    main()

