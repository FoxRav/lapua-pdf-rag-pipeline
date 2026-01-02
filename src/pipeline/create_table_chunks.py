"""Create table-row chunks from parsed table data.

Implements the spec: each row → one chunk with (table_title + header + row).

Usage:
    python -m src.pipeline.create_table_chunks
"""

import json
import sys
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "out" / "2024"


def load_tables(filepath: Path) -> list[dict]:
    """Load tables from PaddleOCR parser output."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("tables", [])


def extract_table_info(table: dict) -> dict[str, Any]:
    """Extract structured info from a table (PaddleOCR format)."""
    page = table.get("page", 0)
    region = table.get("region", 0)
    table_index = table.get("table_index", 0)
    
    # Create table_id
    table_id = f"table_p{page}_r{region}_t{table_index}"
    
    # Get rows from PaddleOCR output
    rows = table.get("rows", [])
    
    # Use markdown for readable text
    markdown = table.get("markdown", "")
    
    # Try to get header (first row)
    header = rows[0] if rows else []
    
    # Try to detect table title
    title = ""
    if rows and len(rows[0]) == 1:
        # Single cell first row might be title
        title = rows[0][0] if rows[0] else ""
        rows = rows[1:]
        header = rows[0] if rows else []
    
    return {
        "table_id": table_id,
        "page": page,
        "region": region,
        "title": title,
        "header": header,
        "rows": rows[1:] if len(rows) > 1 else [],  # Skip header
        "markdown": markdown,
    }


def create_row_chunks(table_info: dict) -> list[dict]:
    """Create one chunk per row with context (title + header + row)."""
    chunks = []
    
    table_id = table_info["table_id"]
    page = table_info["page"]
    title = table_info.get("title", "")
    header = table_info.get("header", [])
    rows = table_info.get("rows", [])
    
    # Clean header
    header_clean = [str(h).strip() for h in header if h]
    header_text = " | ".join(header_clean) if header_clean else ""
    
    for row_idx, row in enumerate(rows):
        # Clean row data
        row_clean = [str(cell).strip() for cell in row if cell]
        row_text = " | ".join(row_clean) if row_clean else ""
        
        if not row_text:
            continue
        
        # Build chunk text: title + header + row
        parts = []
        if title:
            parts.append(f"Taulukko: {title}")
        if header_text:
            parts.append(f"Otsikot: {header_text}")
        parts.append(f"Rivi {row_idx + 1}: {row_text}")
        
        chunk_text = "\n".join(parts)
        
        chunk = {
            "chunk_id": f"{table_id}_row{row_idx}",
            "doc_id": "lapua_tilinpaatos_2024",
            "page": page,
            "element_type": "table_row",
            "table_id": table_id,
            "row_index": row_idx,
            "text": chunk_text,
            "source_refs": [{
                "type": "table",
                "table_id": table_id,
                "page": page,
                "row": row_idx,
            }],
            # Keep raw data for evidence
            "header_raw": header,
            "row_raw": row,
        }
        chunks.append(chunk)
    
    # Also create a summary chunk for the table
    if rows:
        summary_parts = []
        if title:
            summary_parts.append(f"Taulukko: {title}")
        if header_text:
            summary_parts.append(f"Sarakkeet: {header_text}")
        summary_parts.append(f"Rivejä: {len(rows)}")
        summary_parts.append(f"Sivu: {page}")
        
        # Add first few rows as preview
        preview_rows = rows[:3]
        for i, row in enumerate(preview_rows):
            row_clean = [str(cell).strip() for cell in row if cell]
            if row_clean:
                summary_parts.append(f"  Rivi {i+1}: {' | '.join(row_clean[:5])}")
        
        summary_chunk = {
            "chunk_id": f"{table_id}_summary",
            "doc_id": "lapua_tilinpaatos_2024",
            "page": page,
            "element_type": "table_summary",
            "table_id": table_id,
            "text": "\n".join(summary_parts),
            "source_refs": [{
                "type": "table",
                "table_id": table_id,
                "page": page,
            }],
            "row_count": len(rows),
        }
        chunks.append(summary_chunk)
    
    return chunks


def main() -> None:
    print("=" * 60)
    print("CREATING TABLE-ROW CHUNKS")
    print("=" * 60)
    
    # Load tables from PaddleOCR output
    tables_path = DATA_DIR / "tables_from_pdfparser.json"
    
    if not tables_path.exists():
        print(f"❌ Tables file not found: {tables_path}")
        sys.exit(1)
    
    print(f"\n📄 Loading tables from: {tables_path}")
    
    with open(tables_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    tables = data.get("tables", [])
    print(f"📊 Found {len(tables)} tables")
    
    # Process each table
    all_chunks = []
    tables_processed = 0
    
    for table in tables:
        table_info = extract_table_info(table)
        row_chunks = create_row_chunks(table_info)
        all_chunks.extend(row_chunks)
        tables_processed += 1
        
        if row_chunks:
            print(f"  ✅ {table_info['table_id']}: {len(row_chunks)} chunks (page {table_info['page']})")
    
    # Save tables.jsonl
    tables_jsonl_path = DATA_DIR / "tables.jsonl"
    with open(tables_jsonl_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    
    print(f"\n✅ Saved {len(all_chunks)} table chunks to: {tables_jsonl_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Tables processed: {tables_processed}")
    print(f"Total chunks: {len(all_chunks)}")
    
    # Breakdown by type
    row_chunks = [c for c in all_chunks if c["element_type"] == "table_row"]
    summary_chunks = [c for c in all_chunks if c["element_type"] == "table_summary"]
    print(f"  - Row chunks: {len(row_chunks)}")
    print(f"  - Summary chunks: {len(summary_chunks)}")


if __name__ == "__main__":
    main()

