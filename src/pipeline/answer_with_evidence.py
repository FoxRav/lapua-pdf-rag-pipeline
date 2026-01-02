"""LLM answer generation with mandatory evidence.

Implements the spec: no hallucination, numbers must have evidence.

Usage:
    python -m src.pipeline.answer_with_evidence "Mikä on vuosikate?"
"""

import argparse
import json
import pickle
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.pipeline.reranker import Reranker

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent.parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "out" / "complete_index"

# Models
EMBEDDING_MODEL = "BAAI/bge-m3"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_ADAPTER = "CCG-FAKTUM/lapua-llm-v2"


@dataclass
class Evidence:
    """Evidence for an answer."""
    source_doc: str
    page: int
    chunk_type: str
    table_id: str | None
    text_snippet: str
    numbers_found: list[str]


@dataclass
class AnswerResult:
    """Complete answer with evidence."""
    query: str
    answer: str
    evidence: list[Evidence]
    numbers_in_answer: list[str]
    numbers_verified: bool
    confidence: str  # high, medium, low, no_evidence


def extract_numbers(text: str) -> list[str]:
    """Extract numbers from text."""
    # Match numbers with various formats
    patterns = [
        r'-?\d+[\s\xa0]?\d*[\s\xa0]?\d*[,\.]\d+',  # Decimals with spaces
        r'-?\d+[\s\xa0]+\d+[\s\xa0]+\d+',          # Large numbers with spaces
        r'-?\d+[\s\xa0]+\d+',                       # Medium numbers with spaces
        r'-?\d+[,\.]\d+',                           # Decimals
        r'-?\d+',                                    # Integers
    ]
    
    numbers = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for m in matches:
            # Normalize
            normalized = m.replace('\xa0', ' ').strip()
            if normalized and len(normalized) > 0:
                numbers.append(normalized)
    
    return list(set(numbers))


def load_index():
    """Load the complete index."""
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
    
    return chunks, bm25, texts, faiss_index


def search_and_rerank(
    query: str,
    chunks: list,
    bm25,
    texts: list,
    faiss_index,
    embed_model: SentenceTransformer,
    reranker: Reranker,
    top_k: int = 5,
) -> list[dict]:
    """Search with hybrid + rerank."""
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
    
    # Get top candidates for reranking
    top_indices = np.argsort(combined_scores)[::-1][:50]
    candidates = []
    for idx in top_indices:
        chunk = chunks[idx].copy()
        chunk["score"] = float(combined_scores[idx])
        candidates.append(chunk)
    
    # Rerank
    reranked = reranker.rerank(query, candidates, top_k=top_k)
    return [r.chunk for r in reranked]


def create_evidence(chunks: list[dict]) -> list[Evidence]:
    """Create evidence objects from chunks."""
    evidence = []
    for chunk in chunks:
        text = chunk.get("text", "")
        numbers = extract_numbers(text)
        
        evidence.append(Evidence(
            source_doc=chunk.get("source_doc", "unknown"),
            page=chunk.get("page", 0),
            chunk_type=chunk.get("chunk_type", chunk.get("element_type", "unknown")),
            table_id=chunk.get("table_id"),
            text_snippet=text[:500],
            numbers_found=numbers,
        ))
    
    return evidence


def generate_answer(
    query: str,
    evidence: list[Evidence],
    model,
    tokenizer,
) -> str:
    """Generate answer using LLM with evidence-focused prompt."""
    
    # Build context from evidence
    context_parts = []
    for i, e in enumerate(evidence, 1):
        source_info = f"[{i}] {e.source_doc}, sivu {e.page}"
        if e.table_id:
            source_info += f", taulukko {e.table_id}"
        if e.numbers_found:
            source_info += f" (luvut: {', '.join(e.numbers_found[:5])})"
        
        context_parts.append(f"{source_info}:\n{e.text_snippet}")
    
    context = "\n\n".join(context_parts)
    
    # Evidence-focused prompt
    prompt = f"""Olet asiantuntija, joka vastaa kysymyksiin kuntadokumenttien perusteella.

SÄÄNNÖT:
1. Vastaa VAIN annetun kontekstin perusteella
2. Jos et löydä vastausta, sano "Ei löydy asiakirjasta"
3. Käytä TARKKOJA lukuja kontekstista (älä pyöristä)
4. Ilmoita aina lähde (sivu, taulukko)

KONTEKSTI:
{context}

KYSYMYS: {query}

VASTAA seuraavassa muodossa:
VASTAUS: [lyhyt vastaus]
LUVUT: [kaikki vastauksessa käytetyt luvut]
LÄHTEET: [sivu ja taulukko mistä luvut löytyvät]
"""

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True)
    
    return answer


def verify_numbers(answer: str, evidence: list[Evidence]) -> tuple[bool, list[str]]:
    """Verify that numbers in answer appear in evidence."""
    answer_numbers = extract_numbers(answer)
    
    # Collect all numbers from evidence
    evidence_numbers = set()
    for e in evidence:
        for n in e.numbers_found:
            # Normalize for comparison
            normalized = n.replace(' ', '').replace('\xa0', '').replace('.', '').replace(',', '.')
            evidence_numbers.add(normalized)
    
    # Check each answer number
    verified = True
    for n in answer_numbers:
        normalized = n.replace(' ', '').replace('\xa0', '').replace('.', '').replace(',', '.')
        if normalized not in evidence_numbers:
            # Check if it's a simple integer that might be formatted differently
            try:
                num_val = float(normalized)
                found = any(abs(float(en) - num_val) < 0.01 for en in evidence_numbers if en)
                if not found:
                    verified = False
            except ValueError:
                verified = False
    
    return verified, answer_numbers


def answer_question(query: str) -> AnswerResult:
    """Complete pipeline: search → rerank → generate → verify."""
    
    print(f"🔍 Query: {query}")
    print("=" * 70)
    
    # Load index
    print("Loading index...")
    chunks, bm25, texts, faiss_index = load_index()
    
    # Load models
    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL, device="cuda")
    
    print("Loading reranker...")
    reranker = Reranker()
    
    print("Loading LLM...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        load_in_4bit=True,
        trust_remote_code=True,
    )
    
    # Try to load LoRA adapter
    try:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, LORA_ADAPTER)
        print(f"  Loaded LoRA: {LORA_ADAPTER}")
    except Exception as e:
        print(f"  LoRA not loaded: {e}")
    
    # Search and rerank
    print("\nSearching and reranking...")
    top_chunks = search_and_rerank(
        query, chunks, bm25, texts, faiss_index, embed_model, reranker, top_k=5
    )
    
    # Create evidence
    evidence = create_evidence(top_chunks)
    
    print(f"Found {len(evidence)} evidence chunks")
    for i, e in enumerate(evidence, 1):
        print(f"  [{i}] {e.source_doc}, sivu {e.page}, luvut: {e.numbers_found[:3]}")
    
    # Generate answer
    print("\nGenerating answer...")
    answer = generate_answer(query, evidence, model, tokenizer)
    
    # Verify numbers
    verified, answer_numbers = verify_numbers(answer, evidence)
    
    # Determine confidence
    if not evidence:
        confidence = "no_evidence"
    elif verified and answer_numbers:
        confidence = "high"
    elif verified:
        confidence = "medium"
    else:
        confidence = "low"
    
    result = AnswerResult(
        query=query,
        answer=answer,
        evidence=evidence,
        numbers_in_answer=answer_numbers,
        numbers_verified=verified,
        confidence=confidence,
    )
    
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer questions with evidence")
    parser.add_argument("query", type=str, help="Question to answer")
    args = parser.parse_args()
    
    result = answer_question(args.query)
    
    print("\n" + "=" * 70)
    print("ANSWER WITH EVIDENCE")
    print("=" * 70)
    
    print(f"\n{result.answer}")
    
    print(f"\n📊 Numerot vastauksessa: {result.numbers_in_answer}")
    print(f"✅ Numerot verifioitu: {result.numbers_verified}")
    print(f"🎯 Luottamus: {result.confidence}")
    
    print("\n📚 LÄHTEET:")
    for i, e in enumerate(result.evidence, 1):
        table_info = f", taulukko {e.table_id}" if e.table_id else ""
        print(f"  [{i}] {e.source_doc}, sivu {e.page}{table_info}")


if __name__ == "__main__":
    main()

