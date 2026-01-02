"""Cross-encoder reranker for improved retrieval.

Uses a cross-encoder model to rerank initial retrieval results.

Usage:
    from src.pipeline.reranker import Reranker
    reranker = Reranker()
    reranked = reranker.rerank(query, candidates, top_k=10)
"""

import sys
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

sys.stdout.reconfigure(encoding='utf-8')

# Best multilingual reranker models
RERANKER_MODELS = {
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",  # Multilingual, best quality
    "bge-reranker-base": "BAAI/bge-reranker-base",    # English-focused, faster
    "mmarco-mMiniLMv2": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",  # Multilingual
}

DEFAULT_MODEL = "bge-reranker-v2-m3"


@dataclass
class RerankedResult:
    """Result with reranker score."""
    chunk: dict
    rerank_score: float
    original_score: float
    original_rank: int


class Reranker:
    """Cross-encoder reranker for retrieval results."""
    
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cuda"):
        """Initialize reranker with specified model."""
        model_path = RERANKER_MODELS.get(model_name, model_name)
        print(f"Loading reranker: {model_path}")
        self.model = CrossEncoder(model_path, device=device)
        self.model_name = model_name
    
    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 10,
        text_key: str = "text",
    ) -> list[RerankedResult]:
        """Rerank candidates using cross-encoder.
        
        Args:
            query: Search query
            candidates: List of chunk dictionaries with 'text' key
            top_k: Number of results to return after reranking
            text_key: Key to use for candidate text
            
        Returns:
            List of RerankedResult with rerank scores
        """
        if not candidates:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [(query, c.get(text_key, "")) for c in candidates]
        
        # Get cross-encoder scores
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Create results with scores
        results = []
        for i, (chunk, score) in enumerate(zip(candidates, scores)):
            results.append(RerankedResult(
                chunk=chunk,
                rerank_score=float(score),
                original_score=chunk.get("score", 0.0),
                original_rank=i + 1,
            ))
        
        # Sort by rerank score
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        return results[:top_k]


def test_reranker():
    """Quick test of reranker."""
    print("Testing reranker...")
    
    reranker = Reranker()
    
    # Test candidates
    candidates = [
        {"text": "Lapuan kaupungin vuosikate oli 7,5 miljoonaa euroa.", "score": 0.8},
        {"text": "Valtuustossa on 35 jäsentä.", "score": 0.7},
        {"text": "Henkilöstön määrä on 470 vakinaista.", "score": 0.6},
    ]
    
    query = "Mikä on vuosikate?"
    
    results = reranker.rerank(query, candidates, top_k=3)
    
    print(f"\nQuery: {query}")
    print("-" * 50)
    for i, r in enumerate(results, 1):
        print(f"{i}. Rerank: {r.rerank_score:.3f} (was #{r.original_rank})")
        print(f"   {r.chunk['text'][:80]}...")
    
    print("\n✅ Reranker test passed!")


if __name__ == "__main__":
    test_reranker()

