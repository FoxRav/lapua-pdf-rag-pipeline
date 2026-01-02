"""RAG Query Module - Tilinpäätöstietojen hakutoiminnallisuus.

Hybrid search: BM25 + Vector (FAISS) with BGE-M3.
"""

import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from src.common.io import get_output_dir

# Ensure proper UTF-8 output
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

EMBEDDING_MODEL = "BAAI/bge-m3"


@dataclass
class SearchResult:
    """Single search result with source information."""
    
    chunk_id: str
    text: str
    score: float
    page: int | None
    table_id: str | None
    statement_type: str | None
    source_type: str  # "table" or "statement"


class TilinpaatosRAG:
    """RAG-hakukone Lapuan tilinpäätöstiedoille."""
    
    def __init__(self, year: int, device: str = "auto"):
        self.year = year
        self.output_dir = get_output_dir(year)
        self.index_dir = self.output_dir / "index"
        
        # Determine device
        if device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load components
        self._load_index()
        self._load_chunks()
        self._load_embedding_model()
        
        print(f"RAG initialized: {len(self.chunks)} chunks, device={self.device}")
    
    def _load_index(self) -> None:
        """Load FAISS and BM25 indexes."""
        # FAISS
        faiss_path = self.index_dir / "faiss.index"
        self.faiss_index = faiss.read_index(str(faiss_path))
        
        # BM25 (stored as tuple: (bm25_index, texts))
        bm25_path = self.index_dir / "bm25.pkl"
        with open(bm25_path, "rb") as f:
            bm25_data = pickle.load(f)
            if isinstance(bm25_data, tuple):
                self.bm25 = bm25_data[0]
            else:
                self.bm25 = bm25_data
        
        # Metadata
        metadata_path = self.index_dir / "metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.index_metadata = json.load(f)
    
    def _load_chunks(self) -> None:
        """Load chunk metadata and texts."""
        chunks_path = self.index_dir / "chunks_metadata.json"
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        
        # Build text list for BM25
        self.texts = [chunk.get("text", "") for chunk in self.chunks]
    
    def _load_embedding_model(self) -> None:
        """Load BGE-M3 embedding model via sentence-transformers."""
        from sentence_transformers import SentenceTransformer
        
        model_name = self.index_metadata.get("model_name", EMBEDDING_MODEL)
        print(f"Loading embedding model: {model_name} on {self.device}...")
        
        self.embed_model = SentenceTransformer(model_name, device=self.device)
    
    def _embed_query(self, query: str) -> np.ndarray:
        """Embed query text using BGE-M3."""
        embedding = self.embed_model.encode([query], convert_to_numpy=True)
        return embedding.astype("float32")
    
    def search_bm25(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """BM25 sparse search."""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk = self.chunks[idx]
                results.append(self._chunk_to_result(chunk, float(scores[idx])))
        
        return results
    
    def search_vector(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Vector similarity search using FAISS."""
        query_embedding = self._embed_query(query)
        
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                chunk = self.chunks[idx]
                # Convert L2 distance to similarity score (lower is better)
                score = 1.0 / (1.0 + float(dist))
                results.append(self._chunk_to_result(chunk, score))
        
        return results
    
    def search_hybrid(
        self,
        query: str,
        top_k: int = 10,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
    ) -> list[SearchResult]:
        """Hybrid search combining BM25 and vector scores."""
        # Get more candidates for reranking
        bm25_results = self.search_bm25(query, top_k=top_k * 2)
        vector_results = self.search_vector(query, top_k=top_k * 2)
        
        # Combine scores
        scores: dict[str, float] = {}
        chunks_map: dict[str, dict[str, Any]] = {}
        
        # Normalize BM25 scores
        bm25_max = max((r.score for r in bm25_results), default=1.0)
        for result in bm25_results:
            norm_score = result.score / bm25_max if bm25_max > 0 else 0
            scores[result.chunk_id] = scores.get(result.chunk_id, 0) + bm25_weight * norm_score
            chunks_map[result.chunk_id] = {
                "text": result.text,
                "page": result.page,
                "table_id": result.table_id,
                "statement_type": result.statement_type,
                "source_type": result.source_type,
            }
        
        # Normalize vector scores
        vector_max = max((r.score for r in vector_results), default=1.0)
        for result in vector_results:
            norm_score = result.score / vector_max if vector_max > 0 else 0
            scores[result.chunk_id] = scores.get(result.chunk_id, 0) + vector_weight * norm_score
            if result.chunk_id not in chunks_map:
                chunks_map[result.chunk_id] = {
                    "text": result.text,
                    "page": result.page,
                    "table_id": result.table_id,
                    "statement_type": result.statement_type,
                    "source_type": result.source_type,
                }
        
        # Sort by combined score
        sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for chunk_id, score in sorted_chunks:
            info = chunks_map[chunk_id]
            results.append(SearchResult(
                chunk_id=chunk_id,
                text=info["text"],
                score=score,
                page=info["page"],
                table_id=info["table_id"],
                statement_type=info["statement_type"],
                source_type=info["source_type"],
            ))
        
        return results
    
    def _chunk_to_result(self, chunk: dict[str, Any], score: float) -> SearchResult:
        """Convert chunk dict to SearchResult."""
        chunk_id = chunk.get("chunk_id", "")
        
        # Determine source type from chunk_id
        if chunk_id.startswith("statement_"):
            source_type = "statement"
            statement_type = chunk_id.replace("statement_", "").replace(f"_{self.year}", "")
        else:
            source_type = "table"
            statement_type = None
        
        return SearchResult(
            chunk_id=chunk_id,
            text=chunk.get("text", ""),
            score=score,
            page=chunk.get("page"),
            table_id=chunk.get("table_id"),
            statement_type=statement_type,
            source_type=source_type,
        )
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        method: str = "hybrid",
    ) -> list[SearchResult]:
        """Search for relevant chunks.
        
        Args:
            question: Natural language question
            top_k: Number of results
            method: "bm25", "vector", or "hybrid"
        
        Returns:
            List of SearchResult with sources
        """
        if method == "bm25":
            return self.search_bm25(question, top_k)
        elif method == "vector":
            return self.search_vector(question, top_k)
        else:
            return self.search_hybrid(question, top_k)
    
    def format_results(self, results: list[SearchResult]) -> str:
        """Format results for display."""
        output = []
        for i, r in enumerate(results, 1):
            source = f"[Sivu {r.page}]" if r.page else ""
            if r.table_id:
                source += f" [Taulukko: {r.table_id}]"
            if r.statement_type:
                source += f" [{r.statement_type}]"
            
            output.append(f"\n--- Tulos {i} (score: {r.score:.3f}) {source} ---")
            # Truncate long texts and handle encoding
            text = r.text[:500] + "..." if len(r.text) > 500 else r.text
            # Keep Finnish characters, only replace truly problematic ones
            text = text.encode("utf-8", errors="replace").decode("utf-8")
            output.append(text)
        
        return "\n".join(output)


def interactive_query(year: int) -> None:
    """Interactive query mode."""
    print(f"\n=== Lapuan tilinpäätös {year} RAG-haku ===")
    print("Kirjoita kysymys tai 'quit' lopettaaksesi.\n")
    
    rag = TilinpaatosRAG(year)
    
    while True:
        try:
            question = input("\nKysymys: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not question or question.lower() in ("quit", "exit", "q"):
            break
        
        results = rag.query(question, top_k=5)
        print(rag.format_results(results))


def main(year: int, question: str | None = None) -> None:
    """Main entry point."""
    if question:
        # Single query mode
        rag = TilinpaatosRAG(year)
        results = rag.query(question, top_k=5)
        print(rag.format_results(results))
    else:
        # Interactive mode
        interactive_query(year)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline.query YEAR [QUESTION]")
        print("       python -m src.pipeline.query 2024 'Mikä on vuosikate?'")
        sys.exit(1)
    
    year = int(sys.argv[1])
    question = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None
    main(year, question)

