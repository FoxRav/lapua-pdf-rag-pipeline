"""Chunk documents by structure (sections, tables, statements)."""

import sys
from pathlib import Path
from typing import Any

from src.common.io import get_output_dir, read_json, read_jsonl, write_jsonl
from src.common.schema import Document


def create_section_chunks(document: Document) -> list[dict[str, Any]]:
    """Create page-based text chunks from all elements."""
    chunks = []
    
    # Group elements by page
    page_elements: dict[int, list[str]] = {}
    page_element_ids: dict[int, list[str]] = {}
    
    for elem in document.elements:
        if not elem.text or not elem.text.strip():
            continue
        
        page = elem.page or 1
        if page not in page_elements:
            page_elements[page] = []
            page_element_ids[page] = []
        
        page_elements[page].append(elem.text)
        page_element_ids[page].append(elem.element_id)
    
    # Create chunk per page (or combine small pages)
    for page, texts in sorted(page_elements.items()):
        chunk_text = "\n".join(texts)
        if len(chunk_text.strip()) < 50:  # Skip very short pages
            continue
        
        chunks.append(
            {
                "chunk_id": f"text_page_{page}",
                "chunk_type": "text",
                "year": document.year,
                "page": page,
                "section_path": "",
                "text": chunk_text,
                "source_element_ids": page_element_ids[page],
            }
        )
    
    return chunks


def create_table_chunks(document: Document, max_rows: int = 50) -> list[dict[str, Any]]:
    """Create table-based chunks."""
    chunks = []
    
    for table in document.tables:
        # Create full table chunk
        table_text_parts = []
        
        # Add headers
        if table.col_headers:
            table_text_parts.append(" | ".join(table.col_headers))
        
        # Add rows (group by row index)
        rows_dict: dict[int, list[str]] = {}
        for cell in table.cells:
            if cell.r not in rows_dict:
                rows_dict[cell.r] = []
            rows_dict[cell.r].append(cell.text_raw)
        
        row_texts = []
        for r in sorted(rows_dict.keys()):
            row_texts.append(" | ".join(rows_dict[r]))
        
        table_text = "\n".join(table_text_parts + row_texts)
        
        chunks.append(
            {
                "chunk_id": f"table_{table.table_id}",
                "chunk_type": "table",
                "year": document.year,
                "page": table.page,
                "section_path": table.section_path,
                "table_id": table.table_id,
                "bbox": table.bbox.to_list(),
                "text": table_text,
                "num_rows": len(rows_dict),
            }
        )
        
        # If table is large, create row-based chunks
        if len(rows_dict) > max_rows:
            for i in range(0, len(rows_dict), max_rows):
                row_chunk_rows = list(sorted(rows_dict.keys()))[i : i + max_rows]
                row_chunk_text = "\n".join([row_texts[r] for r in row_chunk_rows if r < len(row_texts)])
                
                chunks.append(
                    {
                        "chunk_id": f"table_{table.table_id}_rows_{i}_{i+max_rows}",
                        "chunk_type": "table_rows",
                        "year": document.year,
                        "page": table.page,
                        "section_path": table.section_path,
                        "table_id": table.table_id,
                        "bbox": table.bbox.to_list(),
                        "text": row_chunk_text,
                        "row_range": [i, min(i + max_rows, len(rows_dict))],
                    }
                )
    
    return chunks


def create_statement_chunks(document: Document, line_items_path: Path) -> list[dict[str, Any]]:
    """Create statement-based chunks from extracted line items."""
    chunks = []
    
    if not line_items_path.exists():
        return chunks
    
    # Load line items
    import pandas as pd
    
    line_items_df = pd.read_csv(line_items_path)
    
    if line_items_df.empty:
        return chunks
    
    # Group by statement
    for statement in line_items_df["statement"].unique():
        stmt_items = line_items_df[line_items_df["statement"] == statement]
        
        chunk_text_parts = [f"## {statement}"]
        
        for _, row in stmt_items.iterrows():
            chunk_text_parts.append(f"{row['label_original']}: {row['value_eur']:,.2f} €")
        
        chunk_text = "\n".join(chunk_text_parts)
        
        # Get page range from items
        pages = stmt_items["page"].unique()
        page_range = [int(pages.min()), int(pages.max())] if len(pages) > 1 else [int(pages[0]), int(pages[0])]
        
        chunks.append(
            {
                "chunk_id": f"statement_{statement}_{document.year}",
                "chunk_type": "statement",
                "year": document.year,
                "page": page_range[0] if page_range[0] == page_range[1] else f"{page_range[0]}-{page_range[1]}",
                "page_range": page_range if page_range[0] != page_range[1] else None,
                "section_path": statement,
                "text": chunk_text,
                "statement": statement,
                "num_line_items": len(stmt_items),
                "source_table_ids": stmt_items["source_table_id"].unique().tolist(),
            }
        )
    
    return chunks


def main(year: int) -> None:
    """Main chunk function."""
    output_dir = get_output_dir(year)
    line_items_path = output_dir / "line_items_long.csv"
    
    # Find all document JSON files
    document_files = list(output_dir.glob("document_*.json"))
    if not document_files:
        legacy_path = output_dir / "document.json"
        if legacy_path.exists():
            document_files = [legacy_path]
        else:
            print("No documents found. Run ingest first.")
            return
    
    all_section_chunks = []
    all_table_chunks = []
    all_statement_chunks = []
    
    for doc_path in document_files:
        doc_data = read_json(doc_path)
        document = Document(**doc_data)
        
        print(f"Chunking document {document.doc_id}...")
        
        # Create section chunks
        section_chunks = create_section_chunks(document)
        all_section_chunks.extend(section_chunks)
        print(f"  OK: {len(section_chunks)} section chunks")
        
        # Create table chunks
        table_chunks = create_table_chunks(document, max_rows=50)
        all_table_chunks.extend(table_chunks)
        print(f"  OK: {len(table_chunks)} table chunks")
        
        # Create statement chunks
        statement_chunks = create_statement_chunks(document, line_items_path)
        all_statement_chunks.extend(statement_chunks)
        print(f"  OK: {len(statement_chunks)} statement chunks")
    
    # Save all chunks
    write_jsonl(all_section_chunks, output_dir / "section_chunks.jsonl")
    write_jsonl(all_table_chunks, output_dir / "table_chunks.jsonl")
    write_jsonl(all_statement_chunks, output_dir / "statement_chunks.jsonl")
    
    total = len(all_section_chunks) + len(all_table_chunks) + len(all_statement_chunks)
    print(f"\nChunk complete. Total chunks: {total}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline.03_chunk YEAR")
        sys.exit(1)
    
    year = int(sys.argv[1])
    main(year)

