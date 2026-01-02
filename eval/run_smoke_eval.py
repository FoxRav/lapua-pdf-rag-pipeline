"""Smoke test runner with JSON report output.

Runs all T01-T20 tests and produces eval/smoke_run_YYYYMMDD.json with:
- PASS/FAIL per test
- Evidence: page, table_id, bbox, text/cell
- value_raw and value_normalized for numbers

Usage:
    python -m eval.run_smoke_eval
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent.parent
EVAL_DIR = PROJECT_ROOT / "eval"
DATA_DIR = PROJECT_ROOT / "data" / "out" / "2024"


def load_document() -> dict[str, Any]:
    """Load parsed document."""
    doc_path = DATA_DIR / "document_Lapua-Tilinpaatos-2024.json"
    with open(doc_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_section_chunks() -> list[dict[str, Any]]:
    """Load section chunks."""
    chunks_path = DATA_DIR / "section_chunks.jsonl"
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def load_table_chunks() -> list[dict[str, Any]]:
    """Load table chunks."""
    chunks_path = DATA_DIR / "table_chunks.jsonl"
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def load_ground_truth(full: bool = False) -> list[dict[str, Any]]:
    """Load ground truth.
    
    Args:
        full: If True, load full 50-test suite. If False, load 20-test suite.
    """
    if full:
        gt_path = EVAL_DIR / "smoke_2024_full.json"
        with open(gt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("tests", [])
    else:
        gt_path = EVAL_DIR / "smoke_2024_20q.json"
        with open(gt_path, "r", encoding="utf-8") as f:
            return json.load(f)


def get_page_elements(document: dict[str, Any], page: int) -> list[dict[str, Any]]:
    """Get all elements from a specific page."""
    return [elem for elem in document.get("elements", []) if elem.get("page") == page]


def get_page_text(document: dict[str, Any], page: int) -> str:
    """Get all text from a specific page."""
    elements = get_page_elements(document, page)
    return " ".join(elem.get("text", "") for elem in elements)


def get_chunk_text(chunks: list[dict[str, Any]], page: int) -> str:
    """Get chunk text for a specific page."""
    for chunk in chunks:
        if chunk.get("page") == page:
            return chunk.get("text", "")
    return ""


def get_table_chunks_for_page(table_chunks: list[dict[str, Any]], page: int) -> list[dict[str, Any]]:
    """Get all table chunks for a page."""
    return [c for c in table_chunks if c.get("page") == page]


def normalize_text(text: str) -> str:
    """Normalize text for comparison, including Finnish character variants."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    # Handle Finnish OCR variants (ä→a, ö→o, å→a)
    text = text.replace("ä", "a").replace("ö", "o").replace("å", "a")
    text = text.replace("Ä", "a").replace("Ö", "o").replace("Å", "a")
    return text.strip()


def normalize_expected(expected: str) -> str:
    """Normalize expected value same way as text."""
    return normalize_text(expected)


def contains_text(haystack: str, needle: str) -> bool:
    """Check if haystack contains needle."""
    return normalize_text(needle) in normalize_text(haystack)


def extract_number(text: str, pattern: str) -> Optional[float]:
    """Extract a number from text matching a pattern."""
    # Try to find Finnish format numbers (comma as decimal)
    matches = re.findall(r'-?\d+[,\.]\d+|\d+', text)
    for match in matches:
        normalized = match.replace(",", ".").replace(" ", "")
        try:
            return float(normalized)
        except ValueError:
            continue
    return None


def find_evidence(
    document: dict[str, Any],
    section_chunks: list[dict[str, Any]],
    table_chunks: list[dict[str, Any]],
    page: int,
    search_terms: list[str]
) -> dict[str, Any]:
    """Find evidence for a test on a specific page."""
    evidence = {
        "page": page,
        "found_in": [],
        "element_type": None,
        "table_id": None,
        "bbox": None,
        "text_snippet": None,
        "value_raw": None,
        "value_normalized": None,
    }
    
    # Search in document elements
    for elem in get_page_elements(document, page):
        elem_text = elem.get("text", "")
        for term in search_terms:
            if contains_text(elem_text, term):
                evidence["found_in"].append("document_element")
                evidence["element_type"] = elem.get("element_type", "unknown")
                evidence["bbox"] = elem.get("bbox")
                evidence["text_snippet"] = elem_text[:200]
                break
    
    # Search in section chunks
    chunk_text = get_chunk_text(section_chunks, page)
    for term in search_terms:
        if contains_text(chunk_text, term):
            if "section_chunk" not in evidence["found_in"]:
                evidence["found_in"].append("section_chunk")
            if not evidence["text_snippet"]:
                # Find snippet around the term
                idx = normalize_text(chunk_text).find(normalize_text(term))
                if idx >= 0:
                    start = max(0, idx - 50)
                    end = min(len(chunk_text), idx + len(term) + 100)
                    evidence["text_snippet"] = chunk_text[start:end]
    
    # Search in table chunks
    for tc in get_table_chunks_for_page(table_chunks, page):
        tc_text = tc.get("text", "")
        for term in search_terms:
            if contains_text(tc_text, term):
                if "table_chunk" not in evidence["found_in"]:
                    evidence["found_in"].append("table_chunk")
                evidence["element_type"] = "table"
                evidence["table_id"] = tc.get("table_id")
                if not evidence["text_snippet"]:
                    evidence["text_snippet"] = tc_text[:300]
                break
    
    # Extract numeric values if present
    all_text = get_page_text(document, page) + " " + chunk_text
    for tc in get_table_chunks_for_page(table_chunks, page):
        all_text += " " + tc.get("text", "")
    
    for term in search_terms:
        if re.match(r'^-?\d', term):  # Numeric term
            evidence["value_raw"] = term
            # Try to normalize
            normalized = term.replace(",", ".").replace(" ", "").replace("−", "-")
            try:
                evidence["value_normalized"] = float(normalized)
            except ValueError:
                pass
    
    return evidence


def run_test(
    test: dict[str, Any],
    document: dict[str, Any],
    section_chunks: list[dict[str, Any]],
    table_chunks: list[dict[str, Any]]
) -> dict[str, Any]:
    """Run a single test and return result with evidence."""
    test_id = test["id"]
    severity = test["severity"]
    page = test["pdf_page"]
    match_type = test["match"]
    expected = test.get("expected", "")
    expected_number = test.get("expected_number")
    expected_cells = test.get("expected_cells", [])
    
    result = {
        "id": test_id,
        "severity": severity,
        "expected_page": page,
        "match_type": match_type,
        "expected": expected,
        "expected_number": expected_number,
        "pass": False,
        "evidence": None,
        "error": None,
    }
    
    # Get all text for the page
    all_text = get_page_text(document, page)
    all_text += " " + get_chunk_text(section_chunks, page)
    for tc in get_table_chunks_for_page(table_chunks, page):
        all_text += " " + tc.get("text", "")
    
    # Determine search terms
    search_terms = []
    if expected:
        # Split on newlines and extract key parts
        for part in expected.split("\n"):
            part = part.strip()
            if part:
                search_terms.append(part)
        # Also add any numbers
        numbers = re.findall(r'-?\d+[,\.]\d+|\d+', expected)
        search_terms.extend(numbers)
    if expected_cells:
        search_terms.extend(expected_cells)
    if expected_number is not None:
        search_terms.append(str(expected_number))
    
    # Run the test based on match type
    try:
        if match_type == "exact":
            # Check if exact text is found
            passed = contains_text(all_text, expected)
            result["pass"] = passed
        
        elif match_type == "contains":
            # Check if text contains expected substring
            # For multi-line expected, check main keyword
            main_keyword = expected.split("\n")[0].split()[0] if expected else ""
            passed = any(contains_text(all_text, term) for term in search_terms if term)
            result["pass"] = passed
        
        elif match_type == "table_contains":
            # Check if table cells are found
            passed = all(cell in all_text for cell in expected_cells)
            result["pass"] = passed
        
        # Find evidence
        result["evidence"] = find_evidence(
            document, section_chunks, table_chunks, page, search_terms
        )
        
        # Additional number check
        if expected_number is not None:
            num_str = str(expected_number)
            if num_str not in all_text:
                result["pass"] = False
                result["error"] = f"Expected number {expected_number} not found"
        
    except Exception as e:
        result["error"] = str(e)
        result["pass"] = False
    
    return result


def run_all_tests(full: bool = False) -> dict[str, Any]:
    """Run all smoke tests and return full report.
    
    Args:
        full: If True, run full 50-test suite. If False, run 20-test suite.
    """
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
    must_passed = 0
    must_total = 0
    should_passed = 0
    should_total = 0
    
    for test in ground_truth:
        result = run_test(test, document, section_chunks, table_chunks)
        results.append(result)
        
        status = "✅ PASS" if result["pass"] else "❌ FAIL"
        print(f"{result['id']} [{result['severity']}]: {status}")
        
        if result["severity"] == "MUST":
            must_total += 1
            if result["pass"]:
                must_passed += 1
        else:
            should_total += 1
            if result["pass"]:
                should_passed += 1
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "source_pdf": "Lapua-Tilinpaatos-2024.pdf",
        "ground_truth_file": "eval/smoke_2024_20q.json",
        "summary": {
            "total_tests": len(results),
            "must_passed": must_passed,
            "must_total": must_total,
            "must_pass_rate": f"{must_passed}/{must_total}",
            "should_passed": should_passed,
            "should_total": should_total,
            "should_pass_rate": f"{should_passed}/{should_total}",
            "all_must_passed": must_passed == must_total,
        },
        "results": results,
    }
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"MUST:   {must_passed}/{must_total} {'✅' if must_passed == must_total else '❌'}")
    print(f"SHOULD: {should_passed}/{should_total}")
    print(f"{'='*60}")
    
    return report


def save_report(report: dict[str, Any]) -> Path:
    """Save report to file."""
    date_str = datetime.now().strftime("%Y%m%d")
    output_path = EVAL_DIR / f"smoke_run_{date_str}.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\nReport saved to: {output_path}")
    return output_path


def main() -> int:
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run smoke tests for PDF parser")
    parser.add_argument("--full", action="store_true", 
                        help="Run full 50-test suite instead of basic 20-test suite")
    args = parser.parse_args()
    
    report = run_all_tests(full=args.full)
    save_report(report)
    
    # Return exit code based on MUST tests
    if report["summary"]["all_must_passed"]:
        print("\n✅ All MUST tests passed - CI gate OK")
        return 0
    else:
        print("\n❌ Some MUST tests failed - CI gate FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())

