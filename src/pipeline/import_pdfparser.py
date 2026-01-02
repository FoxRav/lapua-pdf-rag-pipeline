"""Import parsed tables from PDF_Parser project.

PDF_Parser uses PaddleOCR PP-StructureV3 for high-quality table extraction.
This adapter converts that output to our canonical schema.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

from src.common.io import get_output_dir, write_json, write_jsonl
from src.common.ids import table_id
from src.common.schema import BBox, Document, Element, Page, Table, TableCell


def parse_table_markdown(md: str) -> list[list[str]]:
    """Parse markdown table to grid."""
    lines = [l.strip() for l in md.strip().split("\n") if l.strip()]
    grid = []
    
    for line in lines:
        if line.startswith("|") and not re.match(r"^\|[\s\-:|]+\|$", line):
            # Table row
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if cells:
                grid.append(cells)
    
    return grid


def convert_pdfparser_to_document(
    tables_json_path: Path,
    year: int,
    source_pdf: str,
) -> Document:
    """Convert PDF_Parser tables.json to our Document schema."""
    with open(tables_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    pages_data = data.get("pages", [])
    
    # Group by page
    pages_dict: dict[int, dict] = {}
    for page_data in pages_data:
        page_num = page_data.get("page", 1)
        if page_num not in pages_dict:
            pages_dict[page_num] = {
                "page_number": page_num,
                "elements": [],
                "tables": [],
                "text": page_data.get("text", ""),
            }
        
        # Check for table content
        if "tables" in page_data:
            for tbl in page_data["tables"]:
                pages_dict[page_num]["tables"].append(tbl)
    
    # Build Document
    all_tables: list[Table] = []
    all_pages: list[Page] = []
    
    for page_num in sorted(pages_dict.keys()):
        page_info = pages_dict[page_num]
        
        page_elements: list[Element] = []
        page_tables: list[Table] = []
        
        # Extract text as elements
        text = page_info.get("text", "")
        if text:
            # Split into paragraphs
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            
            for i, para in enumerate(paragraphs):
                # Detect headings (short lines, often numbered)
                is_heading = (
                    len(para) < 100 
                    and not para.endswith(".")
                    and (para[0].isupper() or para[0].isdigit())
                )
                
                elem = Element(
                    element_id=f"elem_p{page_num}_{i}",
                    element_type="heading" if is_heading else "paragraph",
                    text=para,
                    page=page_num,
                    bbox=BBox(x0=0, y0=0, x1=595, y1=842),
                    heading_level=2 if is_heading else None,
                    section_path="",
                )
                page_elements.append(elem)
        
        # Process tables from the original structure
        # PDF_Parser stores table markdown in different ways
        
        all_pages.append(Page(
            page_num=page_num,
            width=595,
            height=842,
        ))
    
    # Process tables from the dedicated "tables" array
    tables_data = data.get("tables", [])
    
    for tbl_idx, tbl_data in enumerate(tables_data):
        page_num = tbl_data.get("page", 1)
        
        # Use "rows" array if available, otherwise parse markdown
        rows = tbl_data.get("rows", [])
        if not rows:
            md = tbl_data.get("markdown", "")
            if md:
                rows = parse_table_markdown(md)
        
        if not rows:
            continue
        
        # Filter out empty rows
        rows = [row for row in rows if any(cell.strip() for cell in row)]
        if not rows:
            continue
        
        num_rows = len(rows)
        num_cols = max(len(row) for row in rows) if rows else 0
        
        # Build cells
        cells: list[TableCell] = []
        for row_idx, row in enumerate(rows):
            for col_idx, text in enumerate(row):
                if text.strip():  # Only include non-empty cells
                    cells.append(TableCell(
                        r=row_idx,
                        c=col_idx,
                        text_raw=text.strip(),
                    ))
        
        if not cells:
            continue
        
        # Generate table ID
        tbl_id = table_id(page=page_num, bbox=(0, 0, 595, 842))
        tbl_id = f"{tbl_id}_{tbl_idx}"
        
        table = Table(
            table_id=tbl_id,
            page=page_num,
            bbox=BBox(x0=0, y0=0, x1=595, y1=842),
            section_path="",
            cells=cells,
        )
        
        all_tables.append(table)
    
    # Create document ID
    from src.common.ids import hash_bytes
    doc_id = f"doc_{year}_{hash_bytes(source_pdf.encode())}"
    
    # Collect all elements from pages
    all_elements: list[Element] = []
    for page_info in pages_dict.values():
        # Recreate elements that we built earlier
        text = page_info.get("text", "")
        page_num = page_info["page_number"]
        if text:
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            for i, para in enumerate(paragraphs):
                is_heading = (
                    len(para) < 100 
                    and not para.endswith(".")
                    and len(para) > 0
                    and (para[0].isupper() or para[0].isdigit())
                )
                elem = Element(
                    element_id=f"elem_p{page_num}_{i}",
                    element_type="heading" if is_heading else "paragraph",
                    text=para,
                    page=page_num,
                    bbox=BBox(x0=0, y0=0, x1=595, y1=842),
                    heading_level=2 if is_heading else None,
                    section_path="",
                )
                all_elements.append(elem)
    
    return Document(
        doc_id=doc_id,
        year=year,
        source_pdf=source_pdf,
        pages=all_pages,
        elements=all_elements,
        tables=all_tables,
    )


def import_from_pdfparser(year: int, pdfparser_output_dir: Path) -> None:
    """Import tables from PDF_Parser output directory."""
    output_dir = get_output_dir(year)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find tables.json files
    tables_files = list(pdfparser_output_dir.glob("*.tables.json"))
    
    if not tables_files:
        print(f"No *.tables.json files found in {pdfparser_output_dir}")
        return
    
    print(f"Found {len(tables_files)} tables.json files")
    
    for tables_file in tables_files:
        print(f"\nImporting {tables_file.name}...")
        
        # Infer source PDF name
        pdf_name = tables_file.name.replace(".tables.json", ".pdf")
        
        # Convert to Document
        document = convert_pdfparser_to_document(
            tables_file, year, pdf_name
        )
        
        print(f"  Pages: {len(document.pages)}")
        print(f"  Tables: {len(document.tables)}")
        
        # Save
        doc_stem = tables_file.stem.replace(".tables", "")
        write_json(
            document.model_dump(mode="json"),
            output_dir / f"document_{doc_stem}.json",
        )
        
        # Save tables as CSV
        tables_dir = output_dir / "tables"
        tables_dir.mkdir(exist_ok=True)
        
        for table in document.tables:
            save_table_csv(table, tables_dir / f"{table.table_id}.csv")
        
        print(f"  Saved to {output_dir}")


def save_table_csv(table: Table, path: Path) -> None:
    """Save table as CSV."""
    import csv
    
    # Calculate dimensions from cells
    if not table.cells:
        return
    
    max_row = max(cell.r for cell in table.cells)
    max_col = max(cell.c for cell in table.cells)
    
    # Create grid
    grid = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]
    for cell in table.cells:
        if 0 <= cell.r <= max_row and 0 <= cell.c <= max_col:
            grid[cell.r][cell.c] = cell.text_raw
    
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in grid:
            writer.writerow(row)


def main(year: int) -> None:
    """Main function."""
    # Default PDF_Parser output directory
    pdfparser_dirs = [
        Path(f"F:/-DEV-/PDF_Parser/out/lapua_{year}"),
        Path(f"../PDF_Parser/out/lapua_{year}"),
    ]
    
    for pdfparser_dir in pdfparser_dirs:
        if pdfparser_dir.exists():
            import_from_pdfparser(year, pdfparser_dir)
            return
    
    print("PDF_Parser output directory not found")
    print("Expected one of:", pdfparser_dirs)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline.import_pdfparser YEAR")
        sys.exit(1)
    
    year = int(sys.argv[1])
    main(year)

