"""Batch ingest pipeline for multiple PDFs.

Reads manifest.csv and processes all PDFs through:
1. Parse (pdfplumber + OCR fallback)
2. Normalize
3. Chunk
4. Index (optional, can be done separately)

Usage:
    python -m src.pipeline.batch_ingest data/manifest_25pdf.csv --limit 5
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pdfplumber

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = DATA_DIR / "out"


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    """Load manifest CSV."""
    docs = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            docs.append(row)
    return docs


def parse_pdf(pdf_path: Path, doc_id: str, output_dir: Path) -> dict[str, Any]:
    """Parse a single PDF and save results."""
    start_time = time.time()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    elements = []
    errors = []
    page_count = 0
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    # Extract text
                    text = page.extract_text() or ""
                    
                    elements.append({
                        "element_id": f"elem_p{page_num}_0",
                        "page": page_num,
                        "element_type": "paragraph",
                        "text": text,
                        "bbox": {
                            "x0": 0, "y0": 0,
                            "x1": page.width, "y1": page.height
                        },
                    })
                    
                    # Extract tables
                    tables = page.extract_tables() or []
                    for t_idx, table in enumerate(tables):
                        if table:
                            # Convert table to text
                            table_text = "\n".join(
                                " | ".join(str(cell) if cell else "" for cell in row)
                                for row in table
                            )
                            elements.append({
                                "element_id": f"table_p{page_num}_{t_idx}",
                                "page": page_num,
                                "element_type": "table",
                                "text": table_text,
                                "bbox": {"x0": 0, "y0": 0, "x1": page.width, "y1": page.height},
                                "table_data": table,
                            })
                    
                except Exception as e:
                    errors.append({"page": page_num, "error": str(e)})
    
    except Exception as e:
        errors.append({"page": 0, "error": f"Failed to open PDF: {e}"})
    
    parse_time = time.time() - start_time
    
    # Save document.json
    doc_output = {
        "doc_id": doc_id,
        "source_pdf": str(pdf_path),
        "page_count": page_count,
        "elements": elements,
        "parser_version": "1.0.0",
        "produced_at": datetime.now().isoformat(),
    }
    
    with open(output_dir / "document.json", "w", encoding="utf-8") as f:
        json.dump(doc_output, f, ensure_ascii=False, indent=2)
    
    # Save meta.json
    meta = {
        "doc_id": doc_id,
        "pdf_page_count": page_count,
        "elements_count": len(elements),
        "tables_count": sum(1 for e in elements if e["element_type"] == "table"),
        "parse_time_seconds": parse_time,
        "produced_at": datetime.now().isoformat(),
        "parser_version": "1.0.0",
        "errors": errors,
    }
    
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    return meta


def create_chunks(doc_path: Path, output_dir: Path) -> int:
    """Create chunks from parsed document."""
    with open(doc_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    
    chunks = []
    for elem in doc.get("elements", []):
        text = elem.get("text", "")
        if not text or len(text.strip()) < 10:
            continue
        
        chunk = {
            "chunk_id": f"{doc['doc_id']}_{elem['element_id']}",
            "doc_id": doc["doc_id"],
            "page": elem["page"],
            "element_type": elem["element_type"],
            "text": text,
            "source_refs": [elem["element_id"]],
        }
        chunks.append(chunk)
    
    # Save chunks
    with open(output_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    
    return len(chunks)


def run_batch(manifest_path: Path, limit: int | None = None) -> dict[str, Any]:
    """Run batch processing on all PDFs in manifest."""
    docs = load_manifest(manifest_path)
    
    if limit:
        docs = docs[:limit]
    
    print(f"Processing {len(docs)} PDFs...")
    print("=" * 60)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "manifest": str(manifest_path),
        "docs_total": len(docs),
        "docs_processed": 0,
        "docs_failed": 0,
        "pages_total": 0,
        "chunks_total": 0,
        "total_time_seconds": 0,
        "docs": [],
    }
    
    start_time = time.time()
    
    for i, doc in enumerate(docs, 1):
        doc_id = doc["doc_id"]
        filepath = Path(doc["filepath"])
        
        # Make filepath absolute if relative
        if not filepath.is_absolute():
            filepath = PROJECT_ROOT / filepath
        
        print(f"\n[{i}/{len(docs)}] {doc_id}")
        print(f"  File: {filepath.name}")
        
        if not filepath.exists():
            print(f"  ❌ File not found!")
            results["docs_failed"] += 1
            results["docs"].append({
                "doc_id": doc_id,
                "status": "failed",
                "error": "File not found",
            })
            continue
        
        # Create output directory
        output_dir = OUT_DIR / "parsed" / doc_id
        
        try:
            # Parse
            meta = parse_pdf(filepath, doc_id, output_dir)
            print(f"  ✅ Parsed: {meta['pdf_page_count']} pages, {meta['elements_count']} elements, {meta['tables_count']} tables")
            
            # Create chunks
            chunk_count = create_chunks(output_dir / "document.json", output_dir)
            print(f"  ✅ Chunked: {chunk_count} chunks")
            
            results["docs_processed"] += 1
            results["pages_total"] += meta["pdf_page_count"]
            results["chunks_total"] += chunk_count
            
            results["docs"].append({
                "doc_id": doc_id,
                "status": "success",
                "pages": meta["pdf_page_count"],
                "elements": meta["elements_count"],
                "tables": meta["tables_count"],
                "chunks": chunk_count,
                "parse_time": meta["parse_time_seconds"],
                "errors": meta["errors"],
            })
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results["docs_failed"] += 1
            results["docs"].append({
                "doc_id": doc_id,
                "status": "failed",
                "error": str(e),
            })
    
    results["total_time_seconds"] = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)
    print(f"Processed: {results['docs_processed']}/{results['docs_total']}")
    print(f"Failed:    {results['docs_failed']}")
    print(f"Pages:     {results['pages_total']}")
    print(f"Chunks:    {results['chunks_total']}")
    print(f"Time:      {results['total_time_seconds']:.1f}s")
    
    if results["pages_total"] > 0:
        avg_time = results["total_time_seconds"] / results["pages_total"]
        print(f"Avg time/page: {avg_time:.2f}s")
    
    # Save results
    results_path = OUT_DIR / "batch_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch ingest PDFs")
    parser.add_argument("manifest", type=Path, help="Path to manifest.csv")
    parser.add_argument("--limit", type=int, help="Limit number of PDFs to process")
    args = parser.parse_args()
    
    run_batch(args.manifest, args.limit)


if __name__ == "__main__":
    main()

