"""Run batch questions against the complete index.

This script runs all 90 questions (or a subset) and generates a comprehensive report.

Usage:
    python -m eval.run_questions_batch                    # Run all 90 questions
    python -m eval.run_questions_batch --must-only        # Run only MUST questions (20)
    python -m eval.run_questions_batch --category 1_tuloslaskelma  # Run specific category
    python -m eval.run_questions_batch --limit 10         # Run first 10 questions
"""

import argparse
import json
import pickle
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "out" / "complete_index"
QUESTIONS_FILE = PROJECT_ROOT / "eval" / "questions_full_90.json"

EMBEDDING_MODEL = "BAAI/bge-m3"


@dataclass
class QuestionResult:
    """Result for a single question."""
    question_id: str
    question: str
    category: str
    priority: str
    retrieved_chunks: list[dict]
    top_chunk_text: str
    top_chunk_page: int
    top_chunk_doc: str
    search_time_ms: float
    retrieval_score: float
    has_numbers: bool
    numbers_found: list[str]


def load_index():
    """Load the complete index."""
    print("Loading index...")
    
    with open(INDEX_DIR / "chunks_metadata.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    with open(INDEX_DIR / "bm25.pkl", "rb") as f:
        bm25, texts = pickle.load(f)
    
    faiss_path = INDEX_DIR / "faiss.index"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".index") as tmp:
        tmp_path = tmp.name
    shutil.copy(str(faiss_path), tmp_path)
    faiss_index = faiss.read_index(tmp_path)
    Path(tmp_path).unlink()
    
    print(f"  Loaded {len(chunks)} chunks")
    return chunks, bm25, texts, faiss_index


def load_questions(must_only: bool = False, category: str = None, limit: int = None) -> list[dict]:
    """Load questions from JSON file."""
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    questions = data["questions"]
    
    if must_only:
        questions = [q for q in questions if q.get("priority") == "must"]
    
    if category:
        questions = [q for q in questions if q.get("category") == category]
    
    if limit:
        questions = questions[:limit]
    
    return questions


def extract_numbers(text: str) -> list[str]:
    """Extract numbers from text."""
    import re
    patterns = [
        r'-?\d+[\s\xa0]?\d*[\s\xa0]?\d*[,\.]\d+',
        r'-?\d+[\s\xa0]+\d+[\s\xa0]+\d+',
        r'-?\d+[\s\xa0]+\d+',
        r'-?\d+[,\.]\d+',
        r'-?\d+',
    ]
    
    numbers = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for m in matches:
            normalized = m.replace('\xa0', ' ').strip()
            if normalized and len(normalized) > 0:
                numbers.append(normalized)
    
    return list(set(numbers))


def search_hybrid(
    query: str,
    chunks: list,
    bm25,
    texts: list,
    faiss_index,
    embed_model: SentenceTransformer,
    top_k: int = 5,
) -> tuple[list[dict], float]:
    """Perform hybrid search and return results with time."""
    start = time.time()
    
    # BM25
    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    if bm25_scores.max() > 0:
        bm25_scores = bm25_scores / bm25_scores.max()
    
    # Vector
    query_embedding = embed_model.encode([query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding).astype("float32")
    vector_scores, vector_indices = faiss_index.search(query_embedding, len(chunks))
    
    # Combine
    combined_scores = np.zeros(len(chunks))
    combined_scores += 0.5 * bm25_scores
    for i, idx in enumerate(vector_indices[0]):
        combined_scores[idx] += 0.5 * vector_scores[0][i]
    
    # Get top-k
    top_indices = np.argsort(combined_scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        chunk = chunks[idx].copy()
        chunk["score"] = float(combined_scores[idx])
        results.append(chunk)
    
    elapsed = (time.time() - start) * 1000
    return results, elapsed


def run_question(
    question: dict,
    chunks: list,
    bm25,
    texts: list,
    faiss_index,
    embed_model: SentenceTransformer,
) -> QuestionResult:
    """Run a single question and return result."""
    q_text = question["question"]
    
    results, search_time = search_hybrid(
        q_text, chunks, bm25, texts, faiss_index, embed_model, top_k=5
    )
    
    top_chunk = results[0] if results else {}
    top_text = top_chunk.get("text", "")[:500]
    top_page = top_chunk.get("page", 0)
    top_doc = top_chunk.get("source_doc", "unknown")
    top_score = top_chunk.get("score", 0)
    
    numbers = extract_numbers(top_text)
    
    return QuestionResult(
        question_id=question["id"],
        question=q_text,
        category=question.get("category", "unknown"),
        priority=question.get("priority", "should"),
        retrieved_chunks=[
            {
                "doc": r.get("source_doc", "")[:30],
                "page": r.get("page", 0),
                "score": round(r.get("score", 0), 4),
                "type": r.get("chunk_type", r.get("element_type", "unknown")),
            }
            for r in results
        ],
        top_chunk_text=top_text,
        top_chunk_page=top_page,
        top_chunk_doc=top_doc,
        search_time_ms=round(search_time, 2),
        retrieval_score=round(top_score, 4),
        has_numbers=len(numbers) > 0,
        numbers_found=numbers[:10],
    )


def generate_report(results: list[QuestionResult], total_time: float) -> dict:
    """Generate comprehensive report from results."""
    
    # Group by category
    by_category = {}
    for r in results:
        cat = r.category
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)
    
    # Calculate metrics
    must_results = [r for r in results if r.priority == "must"]
    should_results = [r for r in results if r.priority == "should"]
    
    avg_search_time = sum(r.search_time_ms for r in results) / len(results) if results else 0
    avg_score = sum(r.retrieval_score for r in results) / len(results) if results else 0
    
    has_numbers_count = sum(1 for r in results if r.has_numbers)
    
    # Find best and worst
    sorted_by_score = sorted(results, key=lambda r: r.retrieval_score, reverse=True)
    top_5 = sorted_by_score[:5]
    bottom_5 = sorted_by_score[-5:]
    
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(results),
            "must_questions": len(must_results),
            "should_questions": len(should_results),
            "total_time_seconds": round(total_time, 2),
        },
        "summary": {
            "avg_search_time_ms": round(avg_search_time, 2),
            "avg_retrieval_score": round(avg_score, 4),
            "questions_with_numbers": has_numbers_count,
            "questions_without_numbers": len(results) - has_numbers_count,
        },
        "by_category": {
            cat: {
                "count": len(items),
                "avg_score": round(sum(r.retrieval_score for r in items) / len(items), 4),
                "avg_time_ms": round(sum(r.search_time_ms for r in items) / len(items), 2),
            }
            for cat, items in by_category.items()
        },
        "top_5_retrievals": [
            {"id": r.question_id, "score": r.retrieval_score, "question": r.question[:50]}
            for r in top_5
        ],
        "bottom_5_retrievals": [
            {"id": r.question_id, "score": r.retrieval_score, "question": r.question[:50]}
            for r in bottom_5
        ],
        "results": [asdict(r) for r in results],
    }
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Run batch questions")
    parser.add_argument("--must-only", action="store_true", help="Run only MUST questions")
    parser.add_argument("--category", type=str, help="Run specific category")
    parser.add_argument("--limit", type=int, help="Limit number of questions")
    parser.add_argument("--output", type=str, help="Output file path")
    args = parser.parse_args()
    
    print("=" * 70)
    print("BATCH QUESTION RUNNER")
    print("=" * 70)
    
    # Load data
    chunks, bm25, texts, faiss_index = load_index()
    questions = load_questions(args.must_only, args.category, args.limit)
    
    print(f"  Loaded {len(questions)} questions")
    
    # Load embedding model
    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL, device="cuda")
    
    # Run questions
    print(f"\nRunning {len(questions)} questions...")
    print("-" * 70)
    
    results = []
    start_time = time.time()
    
    for i, q in enumerate(questions, 1):
        result = run_question(q, chunks, bm25, texts, faiss_index, embed_model)
        results.append(result)
        
        # Progress indicator
        status = "✓" if result.retrieval_score > 0.3 else "?"
        print(f"[{i:3}/{len(questions)}] {status} {result.question_id}: {result.question[:40]}... (score: {result.retrieval_score:.3f}, {result.search_time_ms:.0f}ms)")
    
    total_time = time.time() - start_time
    
    # Generate report
    print("\n" + "=" * 70)
    print("GENERATING REPORT")
    print("=" * 70)
    
    report = generate_report(results, total_time)
    
    # Print summary
    print(f"\n📊 YHTEENVETO:")
    print(f"  Kysymyksiä: {report['metadata']['total_questions']}")
    print(f"  MUST: {report['metadata']['must_questions']}")
    print(f"  SHOULD: {report['metadata']['should_questions']}")
    print(f"  Kokonaisaika: {report['metadata']['total_time_seconds']:.1f}s")
    print(f"  Keskimääräinen hakuaika: {report['summary']['avg_search_time_ms']:.1f}ms")
    print(f"  Keskimääräinen score: {report['summary']['avg_retrieval_score']:.4f}")
    print(f"  Kysymyksiä joista löytyi lukuja: {report['summary']['questions_with_numbers']}")
    
    print(f"\n📈 TOP 5 (paras retrieval):")
    for item in report["top_5_retrievals"]:
        print(f"  {item['id']}: {item['score']:.4f} - {item['question']}")
    
    print(f"\n📉 BOTTOM 5 (heikoin retrieval):")
    for item in report["bottom_5_retrievals"]:
        print(f"  {item['id']}: {item['score']:.4f} - {item['question']}")
    
    print(f"\n📁 Kategoriat:")
    for cat, stats in report["by_category"].items():
        print(f"  {cat}: {stats['count']} kpl, avg score: {stats['avg_score']:.4f}")
    
    # Save report
    output_file = args.output or f"eval/questions_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = Path(output_file) if args.output else PROJECT_ROOT / output_file
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Raportti tallennettu: {output_path}")
    
    return report


if __name__ == "__main__":
    main()

