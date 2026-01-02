"""Extract financial statement schema from normalized data."""

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.common.io import get_output_dir, read_json, read_parquet, write_json
from src.common.schema import BBox, Document, LineItem, StatementType
from src.common.text_clean import match_financial_statement_keywords


def load_schema_map() -> dict[str, dict[str, str]]:
    """Load schema mapping from config."""
    config_path = Path("configs/schema_map.yaml")
    if not config_path.exists():
        return {}
    
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def normalize_line_item_key(label: str, statement_type: str, schema_map: dict[str, dict[str, str]]) -> str:
    """Normalize line item label to canonical key."""
    label_lower = label.lower().strip()
    
    # Check schema map
    if statement_type in schema_map:
        for finnish_key, canonical_key in schema_map[statement_type].items():
            if finnish_key.lower() in label_lower:
                return canonical_key
    
    # Fallback: normalize label itself
    normalized = label_lower.replace(" ", "_").replace(",", "").replace(".", "")
    return normalized


def find_statement_tables(document: Document, normalized_tables: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Find tables associated with each financial statement type."""
    statement_tables: dict[str, list[dict[str, Any]]] = {
        stmt.value: [] for stmt in StatementType
    }
    
    # Group tables by section path
    for table in document.tables:
        section_path = table.section_path.lower()
        statement_type = match_financial_statement_keywords(section_path)
        
        if statement_type:
            # Get table data
            table_df = normalized_tables[normalized_tables["table_id"] == table.table_id]
            statement_tables[statement_type].append(
                {
                    "table_id": table.table_id,
                    "page": table.page,
                    "bbox": table.bbox,
                    "section_path": table.section_path,
                    "data": table_df,
                }
            )
    
    # Also check table headers for statement keywords
    for table in document.tables:
        headers_text = " ".join(table.col_headers + table.row_headers).lower()
        statement_type = match_financial_statement_keywords(headers_text)
        
        if statement_type:
            table_id_str = table.table_id
            # Avoid duplicates
            if not any(t["table_id"] == table_id_str for t in statement_tables[statement_type]):
                table_df = normalized_tables[normalized_tables["table_id"] == table_id_str]
                statement_tables[statement_type].append(
                    {
                        "table_id": table_id_str,
                        "page": table.page,
                        "bbox": table.bbox,
                        "section_path": table.section_path,
                        "data": table_df,
                    }
                )
    
    # Check table cell contents for statement keywords (for tables with empty section_path)
    for table in document.tables:
        # Get all cell texts
        table_df = normalized_tables[normalized_tables["table_id"] == table.table_id]
        if table_df.empty:
            continue
        
        # Collect first 20 cells' text for keyword matching
        cell_texts = table_df.head(20)["text_raw"].fillna("").tolist()
        combined_text = " ".join(cell_texts).lower()
        
        # Look for specific statement indicators
        statement_type = None
        if any(kw in combined_text for kw in ["vuosikate", "toimintatuotot", "toimintakulut", "toimintakate"]):
            statement_type = StatementType.INCOME_STATEMENT.value
        elif any(kw in combined_text for kw in ["vastaavaa", "vastattavaa", "pysyvät vastaavat", "vaihtuvat vastaavat"]):
            statement_type = StatementType.BALANCE_SHEET.value
        elif any(kw in combined_text for kw in ["toiminnan rahavirta", "investointien rahavirta", "rahoituksen rahavirta", "lainakannan muutokset"]):
            statement_type = StatementType.CASH_FLOW.value
        elif any(kw in combined_text for kw in ["investointimenot", "nettoinvestoinnit", "rahoitusosuudet investointi"]):
            statement_type = StatementType.INVESTMENTS.value
        
        if statement_type:
            table_id_str = table.table_id
            # Avoid duplicates
            if not any(t["table_id"] == table_id_str for t in statement_tables[statement_type]):
                statement_tables[statement_type].append(
                    {
                        "table_id": table_id_str,
                        "page": table.page,
                        "bbox": table.bbox,
                        "section_path": table.section_path,
                        "data": table_df,
                    }
                )
    
    return statement_tables


def extract_line_items_from_table(
    table_info: dict[str, Any],
    doc_id: str,
    year: int,
    statement_type: str,
    schema_map: dict[str, dict[str, str]],
) -> list[LineItem]:
    """Extract line items from a single table."""
    line_items = []
    table_df = table_info["data"]
    
    if table_df.empty:
        return line_items
    
    # Try to identify label column (usually first column or row headers)
    # For now, assume first column contains labels
    label_col = 0
    value_cols = [c for c in range(1, table_df["c"].max() + 1) if c != label_col]
    
    # Group by row to extract label-value pairs
    for r in table_df["r"].unique():
        row_data = table_df[table_df["r"] == r]
        
        # Get label from first column
        label_cells = row_data[row_data["c"] == label_col]
        if label_cells.empty:
            continue
        
        label_original = " ".join(label_cells["text_raw"].tolist()).strip()
        if not label_original:
            continue
        
        # Get values from other columns (take first non-null value)
        value_cells = row_data[row_data["c"].isin(value_cols)]
        value_cells = value_cells[value_cells["value_eur"].notna()]
        
        if value_cells.empty:
            continue
        
        # Use first numeric value found
        first_value = value_cells.iloc[0]
        value_eur = first_value["value_eur"]
        
        line_item_key = normalize_line_item_key(label_original, statement_type, schema_map)
        
        bbox = BBox.from_list([float(v) for v in first_value["bbox"].split(",")])
        
        line_items.append(
            LineItem(
                year=year,
                doc_id=doc_id,
                statement=StatementType(statement_type),
                line_item_key=line_item_key,
                label_original=label_original,
                value_eur=float(value_eur),
                source_table_id=table_info["table_id"],
                page=table_info["page"],
                bbox=bbox,
            )
        )
    
    return line_items


def main(year: int) -> None:
    """Main extract function."""
    output_dir = get_output_dir(year)
    normalized_tables_path = output_dir / "normalized_tables.parquet"
    
    # Find all document JSON files
    document_files = list(output_dir.glob("document_*.json"))
    if not document_files:
        legacy_path = output_dir / "document.json"
        if legacy_path.exists():
            document_files = [legacy_path]
        else:
            print("No documents found. Run ingest first.")
            return
    
    if not normalized_tables_path.exists():
        print("Missing normalized_tables.parquet. Run normalize first.")
        return
    
    # Load normalized tables (combined from all documents)
    normalized_tables = read_parquet(normalized_tables_path)
    
    # Load schema map
    schema_map = load_schema_map()
    
    all_line_items: list[LineItem] = []
    
    for doc_path in document_files:
        doc_data = read_json(doc_path)
        document = Document(**doc_data)
        
        print(f"Extracting schema from document {document.doc_id}...")
    
        # Find statement tables for this document
        statement_tables = find_statement_tables(document, normalized_tables)
        
        # Extract line items
        for statement_type, tables in statement_tables.items():
            if not tables:
                continue
            
            print(f"  Processing {statement_type}: {len(tables)} table(s)")
            
            for table_info in tables:
                items = extract_line_items_from_table(
                    table_info, document.doc_id, year, statement_type, schema_map
                )
                all_line_items.extend(items)
                print(f"    OK: Extracted {len(items)} line items from {table_info['table_id']}")
    
    # Save financial data as JSON
    financial_data = {
        "year": year,
        "num_documents": len(document_files),
        "statements": {
            stmt.value: [
                item.model_dump(mode="json") for item in all_line_items if item.statement.value == stmt.value
            ]
            for stmt in StatementType
        },
    }
    
    write_json(financial_data, output_dir / f"financial_{year}.json")
    
    # Save line items as CSV
    if all_line_items:
        line_items_df = pd.DataFrame([item.model_dump(mode="json") for item in all_line_items])
        # Convert bbox to string for CSV
        line_items_df["bbox"] = line_items_df["bbox"].apply(
            lambda b: ",".join(str(v) for v in b) if isinstance(b, list) else str(b)
        )
        line_items_df.to_csv(output_dir / "line_items_long.csv", index=False)
        print(f"\n  OK: Saved {len(all_line_items)} line items")
    else:
        # Create empty CSV with correct columns
        line_items_df = pd.DataFrame(
            columns=[
                "year",
                "doc_id",
                "statement",
                "line_item_key",
                "label_original",
                "value_eur",
                "source_table_id",
                "page",
                "bbox",
            ]
        )
        line_items_df.to_csv(output_dir / "line_items_long.csv", index=False)
        print("\n  WARNING: No line items extracted")
    
    # Generate report
    required_statements = ["income_statement", "balance_sheet"]
    found_statements = [stmt for stmt in required_statements if statement_tables.get(stmt)]
    
    report = {
        "doc_id": document.doc_id,
        "year": year,
        "required_statements": required_statements,
        "found_statements": found_statements,
        "missing_statements": [s for s in required_statements if s not in found_statements],
        "total_line_items": len(all_line_items),
        "line_items_by_statement": {
            stmt.value: len([i for i in all_line_items if i.statement.value == stmt.value])
            for stmt in StatementType
        },
    }
    
    write_json(report, output_dir / "extract_report.json")
    print(f"\nExtract complete. Found {len(found_statements)}/{len(required_statements)} required statements")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline.02_extract_schema YEAR")
        sys.exit(1)
    
    year = int(sys.argv[1])
    main(year)

