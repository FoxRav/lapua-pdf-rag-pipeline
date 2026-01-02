"""Normalize numbers, units, and table data."""

import sys
from pathlib import Path

import pandas as pd

from src.common.io import get_output_dir, read_json, read_jsonl, write_json, write_jsonl, write_parquet
from src.common.num_parse import extract_unit_multiplier, normalize_table_value
from src.common.schema import Document, Table, TableCell


def normalize_table(table: Table, doc_id: str, year: int) -> pd.DataFrame:
    """Normalize a single table to DataFrame."""
    rows = []
    
    # Detect unit multiplier from table headers or first rows
    unit_multiplier = None
    header_texts = " ".join(table.col_headers + table.row_headers)
    if header_texts:
        mult, _ = extract_unit_multiplier(header_texts)
        if mult:
            unit_multiplier = mult
    
    # Process each cell
    for cell in table.cells:
        value_raw, value_eur = normalize_table_value(cell.text_raw, unit_multiplier)
        
        rows.append(
            {
                "doc_id": doc_id,
                "year": year,
                "table_id": table.table_id,
                "page": table.page,
                "bbox": ",".join(str(v) for v in table.bbox.to_list()),
                "section_path": table.section_path,
                "r": cell.r,
                "c": cell.c,
                "text_raw": cell.text_raw,
                "value_raw": value_raw,
                "value_eur": value_eur,
                "unit_multiplier": unit_multiplier,
            }
        )
    
    return pd.DataFrame(rows)


def normalize_text_elements(document: Document) -> list[dict[str, any]]:
    """Normalize text elements to JSONL format."""
    normalized = []
    
    for elem in document.elements:
        if elem.element_type.value in ["heading", "paragraph"] and elem.text:
            normalized.append(
                {
                    "doc_id": document.doc_id,
                    "year": document.year,
                    "element_id": elem.element_id,
                    "element_type": elem.element_type.value,
                    "page": elem.page,
                    "bbox": elem.bbox.to_list(),
                    "section_path": elem.section_path,
                    "text": elem.text,
                }
            )
    
    return normalized


def main(year: int) -> None:
    """Main normalize function."""
    output_dir = get_output_dir(year)
    
    # Find all document JSON files
    document_files = list(output_dir.glob("document_*.json"))
    if not document_files:
        # Try legacy single document
        legacy_path = output_dir / "document.json"
        if legacy_path.exists():
            document_files = [legacy_path]
        else:
            print(f"No documents found in {output_dir}")
            print("Run ingest first: make ingest YEAR={year}")
            return
    
    all_table_rows = []
    all_text_elements = []
    all_reports = []
    
    for doc_path in document_files:
        # Load document
        doc_data = read_json(doc_path)
        document = Document(**doc_data)
        
        print(f"Normalizing document {document.doc_id}...")
        
        # Normalize tables
        for table in document.tables:
            df = normalize_table(table, document.doc_id, document.year)
            all_table_rows.append(df)
        
        # Normalize text elements
        normalized_text = normalize_text_elements(document)
        all_text_elements.extend(normalized_text)
        
        print(f"  OK: {len(document.tables)} tables, {len(normalized_text)} text elements")
    
    # Save combined results
    if all_table_rows:
        normalized_tables_df = pd.concat(all_table_rows, ignore_index=True)
        write_parquet(normalized_tables_df, output_dir / "normalized_tables.parquet")
        print(f"\nTotal: Normalized {len(normalized_tables_df)} table cells")
    else:
        normalized_tables_df = pd.DataFrame()
        write_parquet(normalized_tables_df, output_dir / "normalized_tables.parquet")
        print("\nWARNING: No tables found in any document")
    
    write_jsonl(all_text_elements, output_dir / "normalized_text.jsonl")
    print(f"Total: Normalized {len(all_text_elements)} text elements")
    
    # Generate report
    num_cells = len(normalized_tables_df)
    num_numeric = normalized_tables_df["value_eur"].notna().sum() if not normalized_tables_df.empty else 0
    num_unclear = num_cells - num_numeric
    
    report = {
        "year": year,
        "num_documents": len(document_files),
        "num_table_cells": int(num_cells),
        "num_numeric_cells": int(num_numeric),
        "num_unclear_cells": int(num_unclear),
        "num_text_elements": len(all_text_elements),
        "numeric_ratio": float(num_numeric / num_cells) if num_cells > 0 else 0.0,
    }
    
    write_json(report, output_dir / "normalize_report.json")
    print(f"\nNormalize complete. Report: {num_numeric}/{num_cells} numeric cells ({report['numeric_ratio']:.1%})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline.01_normalize YEAR")
        sys.exit(1)
    
    year = int(sys.argv[1])
    main(year)

