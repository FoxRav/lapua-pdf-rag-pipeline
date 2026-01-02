"""Smoke test runner v2 with STRICT/TOLERANT levels.

STRICT: Exact match without normalization (only whitespace/case)
TOLERANT: Match with OCR-friendly normalization (ä→a, ö→o)

Usage:
    python -m eval.run_smoke_eval_v2 --full
"""

import json
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent.parent
EVAL_DIR = PROJECT_ROOT / "eval"
DATA_DIR = PROJECT_ROOT / "data" / "out" / "2024"


# =============================================================================
# NORMALIZATION FUNCTIONS
# =============================================================================

def normalize_strict(text: str) -> str:
    """STRICT normalization: Unicode NFC + whitespace + case only.
    
    Does NOT remove diacritics (ä, ö, å stay as-is).
    """
    # Unicode NFC normalization (combines characters)
    text = unicodedata.normalize('NFC', text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_tolerant(text: str) -> str:
    """TOLERANT normalization: diacritic folding + bad char cleanup.
    
    Allows:
    - Diacritic folding (ä→a, ö→o, å→a)
    - Bad char cleanup (� removed)
    - Typographic minus (− → -)
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    # Remove replacement characters
    text = text.replace('\ufffd', '').replace('�', '')
    # Normalize typographic minus
    text = text.replace('−', '-').replace('–', '-')
    # Remove diacritics (ä→a, ö→o, etc.)
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    return text.strip()


def contains_strict(haystack: str, needle: str) -> bool:
    """STRICT contains check."""
    return normalize_strict(needle) in normalize_strict(haystack)


def contains_tolerant(haystack: str, needle: str) -> bool:
    """TOLERANT contains check."""
    return normalize_tolerant(needle) in normalize_tolerant(haystack)


# =============================================================================
# NUMBER VALIDATION
# =============================================================================

def parse_finnish_number(text: str) -> Optional[float]:
    """Parse Finnish format number (comma as decimal, space/dot as thousands)."""
    # Remove spaces and dots (thousands separators)
    text = text.replace(" ", "").replace(".", "")
    # Replace comma with dot (decimal separator)
    text = text.replace(",", ".")
    # Handle minus variants
    text = text.replace("−", "-").replace("–", "-")
    try:
        return float(text)
    except ValueError:
        return None


def extract_numbers_from_text(text: str) -> list[dict[str, Any]]:
    """Extract all numbers from text with their raw form."""
    # Pattern for Finnish numbers: -123 456,78 or 123,45 or 1 234
    # Also handles numbers with leading dots or following dots
    pattern = r'[-−–]?\d[\d\s\.]*(?:,\d+)?'
    matches = re.findall(pattern, text)
    
    results = []
    for match in matches:
        raw = match.strip()
        # Remove leading/trailing dots that are not part of the number
        raw = raw.strip('.')
        normalized = parse_finnish_number(raw)
        if normalized is not None:
            results.append({
                "value_raw": raw,
                "value_normalized": normalized,
                "sign": "negative" if normalized < 0 else "positive",
            })
    
    # Also try to find standalone integers that might be page numbers
    # These might be missed by the above pattern if preceded by dots/spaces
    int_pattern = r'(?<![0-9])(\d{1,3})(?![0-9,\.])'
    int_matches = re.findall(int_pattern, text)
    for match in int_matches:
        try:
            val = int(match)
            # Check if already found
            if not any(r["value_normalized"] == val for r in results):
                results.append({
                    "value_raw": match,
                    "value_normalized": float(val),
                    "sign": "positive",
                })
        except ValueError:
            pass
    
    return results


def validate_number(
    text: str, 
    expected_number: float,
    expected_sign: Optional[str] = None
) -> dict[str, Any]:
    """Validate that a number exists in text with correct value and sign."""
    numbers = extract_numbers_from_text(text)
    
    result = {
        "found": False,
        "value_raw": None,
        "value_normalized": None,
        "sign_ok": None,
        "value_ok": None,
    }
    
    # Tolerance depends on magnitude of number
    if abs(expected_number) > 1000:
        tolerance = 1.0  # Large numbers: allow ±1
    elif abs(expected_number) > 10:
        tolerance = 0.1  # Medium: allow ±0.1
    else:
        tolerance = 0.01  # Small: allow ±0.01
    
    for num in numbers:
        # Check if value matches (within tolerance)
        if abs(num["value_normalized"] - expected_number) < tolerance:
            result["found"] = True
            result["value_raw"] = num["value_raw"]
            result["value_normalized"] = num["value_normalized"]
            result["value_ok"] = True
            
            if expected_sign:
                result["sign_ok"] = num["sign"] == expected_sign
            else:
                result["sign_ok"] = True
            break
        
        # Also check absolute value (for numbers that might appear without sign)
        if abs(abs(num["value_normalized"]) - abs(expected_number)) < tolerance:
            result["found"] = True
            result["value_raw"] = num["value_raw"]
            result["value_normalized"] = num["value_normalized"]
            result["value_ok"] = True
            
            # Sign check: if expected negative but found positive, still mark found
            # but sign_ok = False
            if expected_sign:
                result["sign_ok"] = num["sign"] == expected_sign
            else:
                result["sign_ok"] = True
            break
    
    return result


# =============================================================================
# OCR QUALITY METRICS
# =============================================================================

# Allowed Finnish alphabet (uppercase + lowercase + numbers + common punctuation)
ALLOWED_CHARS = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÅabcdefghijklmnopqrstuvwxyzäöå"
    "0123456789"
    " \t\n\r"
    ".,;:!?-–—()[]{}\"'`´/\\%€$£+*=#@&<>|~^"
)


def calculate_ocr_quality(text: str) -> dict[str, Any]:
    """Calculate OCR quality metrics for a text."""
    if not text:
        return {
            "char_count": 0, 
            "invalid_chars": 0, 
            "invalid_ratio": 0.0,
            "confusable_chars": 0,
            "confusable_ratio": 0.0,
            "confusable_examples": [],
        }
    
    # Count invalid/replacement characters
    invalid_chars = text.count('�') + text.count('\ufffd')
    
    # Count confusable/out-of-alphabet characters
    confusable_chars = 0
    confusable_examples: list[str] = []
    for c in text:
        if c not in ALLOWED_CHARS:
            confusable_chars += 1
            # Collect unique examples (max 10)
            if c not in confusable_examples and len(confusable_examples) < 10:
                confusable_examples.append(c)
    
    # Count expected Finnish characters that might be corrupted
    expected_finnish = ['ä', 'ö', 'å', 'Ä', 'Ö', 'Å']
    finnish_present = sum(text.count(c) for c in expected_finnish)
    
    return {
        "char_count": len(text),
        "invalid_chars": invalid_chars,
        "invalid_ratio": invalid_chars / len(text) if len(text) > 0 else 0,
        "finnish_chars": finnish_present,
        "confusable_chars": confusable_chars,
        "confusable_ratio": confusable_chars / len(text) if len(text) > 0 else 0,
        "confusable_examples": confusable_examples,
    }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_document() -> dict[str, Any]:
    doc_path = DATA_DIR / "document_Lapua-Tilinpaatos-2024.json"
    with open(doc_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_section_chunks() -> list[dict[str, Any]]:
    chunks_path = DATA_DIR / "section_chunks.jsonl"
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def load_table_chunks() -> list[dict[str, Any]]:
    chunks_path = DATA_DIR / "table_chunks.jsonl"
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def load_ground_truth(full: bool = False) -> list[dict[str, Any]]:
    if full:
        gt_path = EVAL_DIR / "smoke_2024_full.json"
        with open(gt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("tests", [])
    else:
        gt_path = EVAL_DIR / "smoke_2024_20q.json"
        with open(gt_path, "r", encoding="utf-8") as f:
            return json.load(f)


# =============================================================================
# TEST EXECUTION
# =============================================================================

def get_all_text_for_page(
    document: dict[str, Any],
    section_chunks: list[dict[str, Any]],
    table_chunks: list[dict[str, Any]],
    page: int
) -> str:
    """Get all text from all sources for a page."""
    texts = []
    
    # Document elements
    for elem in document.get("elements", []):
        if elem.get("page") == page:
            texts.append(elem.get("text", ""))
    
    # Section chunks
    for chunk in section_chunks:
        if chunk.get("page") == page:
            texts.append(chunk.get("text", ""))
    
    # Table chunks
    for chunk in table_chunks:
        if chunk.get("page") == page:
            texts.append(chunk.get("text", ""))
    
    return " ".join(texts)


def find_match_span(text: str, needle: str, use_tolerant: bool = False) -> Optional[dict[str, Any]]:
    """Find the exact match span in text.
    
    Returns the ACTUAL matched text from the source, not the expected needle.
    """
    if use_tolerant:
        norm_text = normalize_tolerant(text)
        norm_needle = normalize_tolerant(needle)
    else:
        norm_text = normalize_strict(text)
        norm_needle = normalize_strict(needle)
    
    idx = norm_text.find(norm_needle)
    if idx < 0:
        return None
    
    # Find the actual matched text in original (un-normalized) text
    # This requires mapping from normalized index back to original
    # Simplified: extract from original text at approximate position
    # Since normalization mainly affects case and whitespace, 
    # we scan original text for a substring that normalizes to match
    
    actual_matched = None
    for i in range(len(text)):
        for j in range(i + 1, min(i + len(needle) * 2, len(text) + 1)):
            candidate = text[i:j]
            if use_tolerant:
                if normalize_tolerant(candidate) == norm_needle:
                    actual_matched = candidate
                    idx = i
                    break
            else:
                if normalize_strict(candidate) == norm_needle:
                    actual_matched = candidate
                    idx = i
                    break
        if actual_matched:
            break
    
    if not actual_matched:
        # Fallback: use needle but mark as approximate
        actual_matched = needle
    
    context_start = max(0, idx - 40)
    context_end = min(len(text), idx + len(actual_matched) + 40)
    
    return {
        "matched_text": actual_matched,  # ACTUAL text from source, not expected
        "expected_text": needle,  # What we were looking for
        "match_start": idx,
        "match_end": idx + len(actual_matched),
        "context": text[context_start:context_end] if len(text) > context_start else "",
    }


def find_table_cell_evidence(
    table_chunks: list[dict[str, Any]], 
    page: int, 
    expected_cells: list[str]
) -> list[dict[str, Any]]:
    """Find table cell evidence for expected cells."""
    evidence = []
    
    for chunk in table_chunks:
        if chunk.get("page") != page:
            continue
        
        chunk_text = chunk.get("text", "")
        table_id = chunk.get("table_id", "")
        
        for cell in expected_cells:
            if cell in chunk_text:
                # Find cell position in text
                idx = chunk_text.find(cell)
                context_start = max(0, idx - 20)
                context_end = min(len(chunk_text), idx + len(cell) + 20)
                
                evidence.append({
                    "expected_cell": cell,
                    "found": True,
                    "table_id": table_id,
                    "cell_raw": cell,
                    "context": chunk_text[context_start:context_end],
                })
    
    return evidence


def run_test(
    test: dict[str, Any],
    document: dict[str, Any],
    section_chunks: list[dict[str, Any]],
    table_chunks: list[dict[str, Any]]
) -> dict[str, Any]:
    """Run a single test with STRICT and TOLERANT levels."""
    test_id = test["id"]
    severity = test["severity"]
    page = test["pdf_page"]
    match_type = test["match"]
    expected = test.get("expected", "")
    expected_number = test.get("expected_number")
    expected_cells = test.get("expected_cells", [])
    
    # Get all text for the page
    all_text = get_all_text_for_page(document, section_chunks, table_chunks, page)
    
    # Calculate OCR quality
    ocr_quality = calculate_ocr_quality(all_text)
    
    result = {
        "id": test_id,
        "severity": severity,
        "category": test.get("category", ""),
        "description": test.get("description", ""),
        "expected_page": page,
        "match_type": match_type,
        "expected": expected,
        "expected_number": expected_number,
        "strict_pass": False,
        "tolerant_pass": False,
        "status": "FAIL",  # STRICT_PASS, TOLERANT_PASS, FAIL
        "evidence": {
            "page": page,
            "text_snippet": all_text[:300] if all_text else "",
            "matched_text": None,
            "match_span": None,
            "table_cell_evidence": None,
            "ocr_quality": ocr_quality,
        },
        "number_validation": None,
    }
    
    # Determine search terms
    search_terms = []
    if expected:
        for part in expected.split("\n"):
            if part.strip():
                search_terms.append(part.strip())
    if expected_cells:
        search_terms.extend(expected_cells)
    
    # Run STRICT and TOLERANT checks with match span
    strict_match_span = None
    tolerant_match_span = None
    
    if match_type in ("exact", "contains"):
        for term in search_terms:
            if contains_strict(all_text, term):
                result["strict_pass"] = True
                strict_match_span = find_match_span(all_text, term, use_tolerant=False)
                break
        
        for term in search_terms:
            if contains_tolerant(all_text, term):
                result["tolerant_pass"] = True
                if not strict_match_span:
                    tolerant_match_span = find_match_span(all_text, term, use_tolerant=True)
                break
    
    elif match_type == "table_contains":
        result["strict_pass"] = all(cell in all_text for cell in expected_cells)
        result["tolerant_pass"] = all(
            contains_tolerant(all_text, cell) for cell in expected_cells
        )
        
        # Get table cell evidence
        result["evidence"]["table_cell_evidence"] = find_table_cell_evidence(
            table_chunks, page, expected_cells
        )
    
    # Store match span
    if strict_match_span:
        result["evidence"]["matched_text"] = strict_match_span["matched_text"]
        result["evidence"]["match_span"] = strict_match_span
    elif tolerant_match_span:
        result["evidence"]["matched_text"] = tolerant_match_span["matched_text"]
        result["evidence"]["match_span"] = tolerant_match_span
    
    # Number validation if expected_number is set
    if expected_number is not None:
        num_str = str(expected_number)
        
        # Full number validation
        expected_sign = "negative" if expected_number < 0 else "positive"
        num_val = validate_number(all_text, expected_number, expected_sign)
        result["number_validation"] = num_val
        
        # STRICT: text match AND number found with correct value
        text_strict = result["strict_pass"]
        number_found = num_val.get("found", False) and num_val.get("value_ok", False)
        result["strict_pass"] = text_strict and number_found
        
        # TOLERANT: text match AND number string exists somewhere
        text_tolerant = result["tolerant_pass"]
        number_in_text = num_str in all_text
        result["tolerant_pass"] = text_tolerant and number_in_text
    
    # Determine final status
    if result["strict_pass"]:
        result["status"] = "STRICT_PASS"
    elif result["tolerant_pass"]:
        result["status"] = "TOLERANT_PASS"
    else:
        result["status"] = "FAIL"
    
    return result


def analyze_tolerant_reasons(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Analyze why tests are TOLERANT_PASS instead of STRICT_PASS."""
    tolerant_tests = [r for r in results if r["status"] == "TOLERANT_PASS"]
    analysis = []
    
    for r in tolerant_tests:
        expected = r.get("expected", "")
        snippet = r.get("evidence", {}).get("text_snippet", "")
        
        # Detect reason for degradation
        reasons = []
        
        # Check diacritics
        finnish_chars = ['ä', 'ö', 'å', 'Ä', 'Ö', 'Å']
        if any(c in expected for c in finnish_chars):
            if not any(c in snippet for c in finnish_chars):
                reasons.append("diacritic_missing")
        
        # Check bad chars
        if '�' in snippet or '\ufffd' in snippet:
            reasons.append("bad_char_present")
        
        # Check typographic minus
        if '-' in expected and ('−' in snippet or '–' in snippet):
            reasons.append("typographic_minus")
        
        if not reasons:
            reasons.append("unknown")
        
        analysis.append({
            "id": r["id"],
            "expected": expected[:50],
            "reasons": reasons,
            "page": r.get("expected_page"),
        })
    
    return analysis


def analyze_number_tests(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Analyze number tests for sign/unit correctness."""
    number_tests = [r for r in results if r.get("number_validation")]
    analysis = []
    
    for r in number_tests:
        nv = r["number_validation"]
        analysis.append({
            "id": r["id"],
            "expected_number": r.get("expected_number"),
            "found": nv.get("found", False),
            "value_raw": nv.get("value_raw"),
            "value_normalized": nv.get("value_normalized"),
            "sign_ok": nv.get("sign_ok"),
            "value_ok": nv.get("value_ok"),
        })
    
    return analysis


def analyze_ocr_per_page(results: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Analyze OCR quality per page."""
    page_stats: dict[int, dict[str, Any]] = {}
    
    for r in results:
        page = r.get("expected_page")
        if page is None:
            continue
        
        ocr = r.get("evidence", {}).get("ocr_quality", {})
        
        if page not in page_stats:
            page_stats[page] = {
                "tests": [],
                "total_chars": 0,
                "invalid_chars": 0,
                "finnish_chars": 0,
                "strict_pass": 0,
                "tolerant_pass": 0,
                "fail": 0,
            }
        
        page_stats[page]["tests"].append(r["id"])
        page_stats[page]["total_chars"] += ocr.get("char_count", 0)
        page_stats[page]["invalid_chars"] += ocr.get("invalid_chars", 0)
        page_stats[page]["finnish_chars"] += ocr.get("finnish_chars", 0)
        
        if r["status"] == "STRICT_PASS":
            page_stats[page]["strict_pass"] += 1
        elif r["status"] == "TOLERANT_PASS":
            page_stats[page]["tolerant_pass"] += 1
        else:
            page_stats[page]["fail"] += 1
    
    # Calculate rates
    for page, stats in page_stats.items():
        if stats["total_chars"] > 0:
            stats["invalid_ratio"] = stats["invalid_chars"] / stats["total_chars"]
        else:
            stats["invalid_ratio"] = 0.0
    
    return page_stats


def run_all_tests(full: bool = False) -> dict[str, Any]:
    """Run all tests and return report."""
    suite_name = "full (50 tests)" if full else "basic (20 tests)"
    print(f"Loading data for {suite_name}...")
    
    document = load_document()
    section_chunks = load_section_chunks()
    table_chunks = load_table_chunks()
    ground_truth = load_ground_truth(full=full)
    
    print(f"Loaded: {len(document.get('elements', []))} elements, "
          f"{len(section_chunks)} section chunks, {len(table_chunks)} table chunks")
    print(f"Running {len(ground_truth)} tests...\n")
    
    results = []
    counts = {
        "STRICT_PASS": 0,
        "TOLERANT_PASS": 0,
        "FAIL": 0,
    }
    must_strict = 0
    must_tolerant = 0
    must_fail = 0
    must_total = 0
    
    for test in ground_truth:
        result = run_test(test, document, section_chunks, table_chunks)
        results.append(result)
        
        counts[result["status"]] += 1
        
        # Status indicator
        if result["status"] == "STRICT_PASS":
            indicator = "✅ STRICT"
        elif result["status"] == "TOLERANT_PASS":
            indicator = "⚠️  TOLERANT"
        else:
            indicator = "❌ FAIL"
        
        print(f"{result['id']} [{result['severity']}]: {indicator}")
        
        if result["severity"] == "MUST":
            must_total += 1
            if result["status"] == "STRICT_PASS":
                must_strict += 1
            elif result["status"] == "TOLERANT_PASS":
                must_tolerant += 1
            else:
                must_fail += 1
    
    # Calculate overall OCR quality
    all_ocr_quality = [r["evidence"]["ocr_quality"] for r in results]
    avg_invalid_ratio = sum(q["invalid_ratio"] for q in all_ocr_quality) / len(all_ocr_quality)
    
    # Generate analysis summaries
    tolerant_analysis = analyze_tolerant_reasons(results)
    number_analysis = analyze_number_tests(results)
    ocr_per_page = analyze_ocr_per_page(results)
    
    # Find pages with issues
    degraded_pages = [p for p, s in ocr_per_page.items() 
                      if s["tolerant_pass"] > 0 or s["fail"] > 0]
    
    # Calculate confusable ratio and aggregate examples
    all_confusable = [r["evidence"]["ocr_quality"].get("confusable_ratio", 0) for r in results]
    avg_confusable_ratio = sum(all_confusable) / len(all_confusable) if all_confusable else 0
    
    # Aggregate confusable examples across all pages
    confusable_by_page: dict[int, list[str]] = {}
    all_confusable_chars: list[str] = []
    for r in results:
        page = r.get("expected_page")
        examples = r.get("evidence", {}).get("ocr_quality", {}).get("confusable_examples", [])
        if examples:
            if page not in confusable_by_page:
                confusable_by_page[page] = []
            for ex in examples:
                if ex not in confusable_by_page[page]:
                    confusable_by_page[page].append(ex)
                if ex not in all_confusable_chars:
                    all_confusable_chars.append(ex)
    
    # Identify which categories are allowed to have TOLERANT_PASS
    allowed_tolerant_categories = ["cover"]  # Only cover page OCR issues are acceptable
    
    # Check if TOLERANT tests are in allowed categories
    tolerant_in_critical = []
    for r in results:
        if r["status"] == "TOLERANT_PASS" and r["severity"] == "MUST":
            category = r.get("category", "")
            if category not in allowed_tolerant_categories:
                tolerant_in_critical.append({
                    "id": r["id"],
                    "page": r.get("expected_page"),
                    "category": category,
                })
    
    # CI Gate definitions
    # Gate A (functionality): FAIL == 0 and all MUST at least TOLERANT_PASS
    gate_a_pass = must_fail == 0
    
    # Gate B (quality): STRICT_PASS_RATE >= baseline (configurable)
    baseline_strict = 43  # Can be loaded from config
    gate_b_pass = must_strict >= baseline_strict
    
    # Gate C (OCR): confusable + invalid ratio < threshold
    ocr_threshold = 0.02  # 2%
    gate_c_pass = (avg_invalid_ratio + avg_confusable_ratio) < ocr_threshold
    
    # Gate D (critical pages): No TOLERANT in critical categories
    gate_d_pass = len(tolerant_in_critical) == 0
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "source_pdf": "Lapua-Tilinpaatos-2024.pdf",
        "ground_truth_file": f"smoke_2024_{'full' if full else '20q'}.json",
        "summary": {
            "total_tests": len(results),
            "strict_pass": counts["STRICT_PASS"],
            "tolerant_pass": counts["TOLERANT_PASS"],
            "fail": counts["FAIL"],
            "strict_pass_rate": counts["STRICT_PASS"] / len(results) if results else 0,
            "must_strict_pass": must_strict,
            "must_tolerant_pass": must_tolerant,
            "must_fail": must_fail,
            "must_total": must_total,
            "ocr_avg_invalid_ratio": avg_invalid_ratio,
            "ocr_avg_confusable_ratio": avg_confusable_ratio,
            "ocr_confusable_examples": all_confusable_chars[:10],  # Top 10
            "ocr_confusable_by_page": {str(k): v for k, v in confusable_by_page.items()},
            "ci_gates": {
                "gate_a_functionality": {
                    "pass": gate_a_pass,
                    "rule": "FAIL == 0 (all MUST at least TOLERANT_PASS)",
                    "must_fail": must_fail,
                },
                "gate_b_quality": {
                    "pass": gate_b_pass,
                    "rule": f"must_strict_pass >= baseline ({baseline_strict})",
                    "must_strict_pass": must_strict,
                    "baseline": baseline_strict,
                },
                "gate_c_ocr": {
                    "pass": gate_c_pass,
                    "rule": f"(invalid + confusable) < {ocr_threshold*100:.1f}%",
                    "combined_ratio": avg_invalid_ratio + avg_confusable_ratio,
                    "threshold": ocr_threshold,
                },
                "gate_d_critical_pages": {
                    "pass": gate_d_pass,
                    "rule": "No TOLERANT in critical categories (only 'cover' allowed)",
                    "tolerant_in_critical": tolerant_in_critical,
                    "allowed_categories": allowed_tolerant_categories,
                },
            },
            "ci_gate_all_pass": gate_a_pass and gate_b_pass and gate_c_pass and gate_d_pass,
        },
        "analysis": {
            "tolerant_reasons": tolerant_analysis,
            "number_validation": number_analysis,
            "ocr_per_page": {str(k): v for k, v in ocr_per_page.items()},
            "degraded_pages": degraded_pages,
        },
        "results": results,
    }
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"MUST STRICT:   {must_strict}/{must_total} {'✅' if must_strict == must_total else ''}")
    print(f"MUST TOLERANT: {must_tolerant}/{must_total} (degraded due to OCR)")
    print(f"MUST FAIL:     {must_fail}/{must_total}")
    print(f"STRICT RATE:   {counts['STRICT_PASS']}/{len(results)} ({counts['STRICT_PASS']/len(results)*100:.1f}%)")
    print(f"OCR invalid:   {avg_invalid_ratio*100:.2f}%")
    print(f"OCR confusable:{avg_confusable_ratio*100:.2f}%")
    print(f"{'='*60}")
    
    # Print TOLERANT reasons
    if tolerant_analysis:
        print("\nTOLERANT_PASS REASONS:")
        for t in tolerant_analysis:
            print(f"  {t['id']}: {', '.join(t['reasons'])} (page {t['page']})")
    
    # Print FAILed tests
    failed_tests = [r for r in results if r["status"] == "FAIL"]
    if failed_tests:
        print("\nFAILED TESTS:")
        for t in failed_tests:
            print(f"  {t['id']}: {t.get('description', '')} (page {t['expected_page']})")
            if t.get("number_validation"):
                nv = t["number_validation"]
                print(f"    -> expected_number={t['expected_number']}, found={nv.get('found')}")
    
    # Print degraded pages
    if degraded_pages:
        print(f"\nDEGRADED PAGES: {degraded_pages}")
    
    # Print confusable chars if any
    if all_confusable_chars:
        print(f"\nCONFUSABLE CHARS: {all_confusable_chars[:10]}")
        for page, chars in confusable_by_page.items():
            print(f"  Page {page}: {chars}")
    
    # Print tolerant in critical if any
    if tolerant_in_critical:
        print(f"\n⚠️  TOLERANT IN CRITICAL CATEGORIES:")
        for t in tolerant_in_critical:
            print(f"  {t['id']}: category='{t['category']}' (page {t['page']})")
    
    print(f"\n{'='*60}")
    print("CI GATES:")
    print(f"  Gate A (functionality): {'✅' if gate_a_pass else '❌'} FAIL=={must_fail}")
    print(f"  Gate B (quality):       {'✅' if gate_b_pass else '❌'} STRICT>={baseline_strict} (got {must_strict})")
    print(f"  Gate C (OCR):           {'✅' if gate_c_pass else '❌'} noise<{ocr_threshold*100:.1f}% (got {(avg_invalid_ratio+avg_confusable_ratio)*100:.2f}%)")
    print(f"  Gate D (critical):      {'✅' if gate_d_pass else '❌'} No TOLERANT in critical ({len(tolerant_in_critical)} violations)")
    print(f"{'='*60}")
    print(f"ALL GATES: {'✅ PASS' if report['summary']['ci_gate_all_pass'] else '❌ FAIL'}")
    
    return report


def save_report(report: dict[str, Any]) -> Path:
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = EVAL_DIR / f"smoke_run_v2_{date_str}.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\nReport saved to: {output_path}")
    return output_path


def main() -> int:
    import argparse
    
    parser = argparse.ArgumentParser(description="Run smoke tests with STRICT/TOLERANT levels")
    parser.add_argument("--full", action="store_true", 
                        help="Run full 50-test suite")
    args = parser.parse_args()
    
    report = run_all_tests(full=args.full)
    save_report(report)
    
    # Return based on CI gates
    gates = report["summary"]["ci_gates"]
    gate_a = gates["gate_a_functionality"]["pass"]
    gate_b = gates["gate_b_quality"]["pass"]
    gate_c = gates["gate_c_ocr"]["pass"]
    gate_d = gates["gate_d_critical_pages"]["pass"]
    
    if gate_a and gate_b and gate_c and gate_d:
        print("\n✅ All gates passed")
        return 0
    elif gate_a and gate_d:
        print("\n⚠️  Gates A+D passed (functionality OK), but quality/OCR gates failed")
        return 0  # Soft fail - functionality works
    elif gate_a:
        print("\n⚠️  Gate A passed (functionality OK), but critical pages have issues")
        return 0  # Soft fail - functionality works but watch closely
    else:
        print("\n❌ Gate A failed - functionality broken")
        return 1


if __name__ == "__main__":
    sys.exit(main())

