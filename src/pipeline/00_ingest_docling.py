"""Ingest PDF documents using pdfplumber + RapidOCR for scanned PDFs."""

import sys
from pathlib import Path
from typing import Any

import pdfplumber
from PIL import Image
import io

from src.common.io import get_raw_dir, get_output_dir, write_json
from src.common.ids import doc_id_from_pdf, element_id, table_id
from src.common.schema import BBox, Document, Element, ElementType, Page, Table, TableCell

# Try to import RapidOCR for scanned PDFs
# Note: RapidOCR uses CPU but is optimized. GPU acceleration is used for embeddings.
try:
    from rapidocr import RapidOCR
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Check PyTorch CUDA for other operations
import torch
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    print(f"PyTorch CUDA available: {torch.cuda.get_device_name(0)}")


def process_with_ocr(pdf_path: Path, year: int) -> tuple[Document, dict[str, Any]]:
    """Process scanned PDF with RapidOCR."""
    doc_id = doc_id_from_pdf(pdf_path, year)
    
    pages: list[Page] = []
    elements: list[Element] = []
    tables: list[Table] = []
    failed_pages: list[int] = []
    
    # Initialize OCR
    # Note: RapidOCR uses onnxruntime which can use GPU if available
    import onnxruntime as ort
    providers = ort.get_available_providers()
    use_gpu = "CUDAExecutionProvider" in providers
    print(f"    OCR using: {'GPU (CUDA)' if use_gpu else 'CPU'}")
    
    ocr = RapidOCR()
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"    Processing {total_pages} pages with OCR...")
        
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                # Extract page info
                pages.append(
                    Page(
                        page_num=page_num,
                        width=float(page.width),
                        height=float(page.height),
                    )
                )
                
                # Convert page to image for OCR
                img = page.to_image(resolution=150)
                img_pil = img.original
                
                # Run OCR
                result = ocr(img_pil)
                
                # RapidOCR returns RapidOCROutput object
                if result and result.boxes is not None and result.txts is not None:
                    # Group OCR results by approximate vertical position
                    lines: dict[int, list[tuple[str, list[float]]]] = {}
                    
                    for bbox_coords, text in zip(result.boxes, result.txts):
                        # Use top-left y coordinate as line key (rounded to 10px)
                        y_key = int(bbox_coords[0][1] // 10) * 10
                        
                        if y_key not in lines:
                            lines[y_key] = []
                        
                        # Get bounding box
                        x_coords = [p[0] for p in bbox_coords]
                        y_coords = [p[1] for p in bbox_coords]
                        bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                        
                        lines[y_key].append((text, bbox))
                    
                    # Sort lines by y position
                    sorted_lines = sorted(lines.items())
                    
                    # Create elements from OCR text
                    current_paragraph: list[str] = []
                    current_bbox = [0, 0, float(page.width), 0]
                    
                    for y_pos, line_items in sorted_lines:
                        # Sort items in line by x position
                        line_items.sort(key=lambda x: x[1][0])
                        line_text = " ".join([item[0] for item in line_items])
                        
                        # Detect headings (uppercase, short lines, etc.)
                        is_heading = (
                            len(line_text) < 80 and 
                            (line_text.isupper() or 
                             line_text.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")) or
                             any(kw in line_text.upper() for kw in ["TULOSLASKELMA", "TASE", "RAHOITUSLASKELMA", "INVESTOINNIT", "LIITETIEDOT"]))
                        )
                        
                        if is_heading:
                            # Save current paragraph if exists
                            if current_paragraph:
                                para_text = " ".join(current_paragraph)
                                para_bbox = BBox(x0=current_bbox[0], y0=current_bbox[1], x1=current_bbox[2], y1=current_bbox[3])
                                elem_id_str = element_id("paragraph", page_num, para_bbox.to_list())
                                elements.append(
                                    Element(
                                        element_id=elem_id_str,
                                        element_type=ElementType.PARAGRAPH,
                                        page=page_num,
                                        bbox=para_bbox,
                                        section_path="",
                                        text=para_text,
                                    )
                                )
                                current_paragraph = []
                            
                            # Create heading element
                            head_bbox = BBox(x0=0, y0=y_pos, x1=float(page.width), y1=y_pos+20)
                            elem_id_str = element_id("heading", page_num, head_bbox.to_list())
                            elements.append(
                                Element(
                                    element_id=elem_id_str,
                                    element_type=ElementType.HEADING,
                                    page=page_num,
                                    bbox=head_bbox,
                                    section_path="",
                                    text=line_text,
                                )
                            )
                        else:
                            current_paragraph.append(line_text)
                            if not current_bbox[1]:
                                current_bbox[1] = y_pos
                            current_bbox[3] = y_pos + 20
                    
                    # Save last paragraph
                    if current_paragraph:
                        para_text = " ".join(current_paragraph)
                        para_bbox = BBox(x0=current_bbox[0], y0=current_bbox[1], x1=current_bbox[2], y1=current_bbox[3])
                        elem_id_str = element_id("paragraph", page_num, para_bbox.to_list())
                        elements.append(
                            Element(
                                element_id=elem_id_str,
                                element_type=ElementType.PARAGRAPH,
                                page=page_num,
                                bbox=para_bbox,
                                section_path="",
                                text=para_text,
                            )
                        )
                
                # Try to detect tables (look for grid patterns)
                # This is a simplified approach - for better table detection,
                # we would need more sophisticated algorithms
                page_tables = page.extract_tables()
                if page_tables:
                    for table_idx, table_data in enumerate(page_tables):
                        if not table_data:
                            continue
                        
                        tbl_bbox = BBox(x0=0, y0=0, x1=float(page.width), y1=float(page.height))
                        tbl_id = table_id(page_num, tbl_bbox.to_list())
                        
                        cells: list[TableCell] = []
                        row_headers: list[str] = []
                        col_headers: list[str] = []
                        
                        for r, row in enumerate(table_data):
                            for c, cell_text in enumerate(row):
                                text_raw = str(cell_text) if cell_text is not None else ""
                                cells.append(TableCell(r=r, c=c, text_raw=text_raw))
                                
                                if r == 0:
                                    col_headers.append(text_raw)
                                if c == 0:
                                    row_headers.append(text_raw)
                        
                        tables.append(
                            Table(
                                table_id=tbl_id,
                                page=page_num,
                                bbox=tbl_bbox,
                                section_path="",
                                cells=cells,
                                row_headers=row_headers,
                                col_headers=col_headers,
                            )
                        )
                
                if page_num % 10 == 0:
                    print(f"    Processed page {page_num}/{total_pages}")
                    
            except Exception as e:
                print(f"    Warning: Failed to process page {page_num}: {e}")
                failed_pages.append(page_num)
    
    document = Document(
        doc_id=doc_id,
        year=year,
        source_pdf=str(pdf_path),
        pages=pages,
        elements=elements,
        tables=tables,
    )
    
    report = {
        "doc_id": doc_id,
        "year": year,
        "source_pdf": str(pdf_path),
        "num_pages": len(pages),
        "num_elements": len(elements),
        "num_tables": len(tables),
        "failed_pages": failed_pages,
        "converter": "pdfplumber+rapidocr",
    }
    
    return document, report


def process_native_pdf(pdf_path: Path, year: int) -> tuple[Document, dict[str, Any]]:
    """Process native (non-scanned) PDF with pdfplumber."""
    doc_id = doc_id_from_pdf(pdf_path, year)
    
    pages: list[Page] = []
    elements: list[Element] = []
    tables: list[Table] = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            pages.append(
                Page(
                    page_num=page_num,
                    width=float(page.width),
                    height=float(page.height),
                )
            )
            
            # Extract text
            text = page.extract_text()
            if text and text.strip():
                bbox = BBox(x0=0, y0=0, x1=float(page.width), y1=float(page.height))
                elem_id_str = element_id("paragraph", page_num, bbox.to_list())
                elements.append(
                    Element(
                        element_id=elem_id_str,
                        element_type=ElementType.PARAGRAPH,
                        page=page_num,
                        bbox=bbox,
                        section_path="",
                        text=text.strip(),
                    )
                )
            
            # Extract tables
            page_tables = page.extract_tables()
            for table_idx, table_data in enumerate(page_tables):
                if not table_data:
                    continue
                
                tbl_bbox = BBox(x0=0, y0=0, x1=float(page.width), y1=float(page.height))
                tbl_id = table_id(page_num, tbl_bbox.to_list())
                
                cells: list[TableCell] = []
                row_headers: list[str] = []
                col_headers: list[str] = []
                
                for r, row in enumerate(table_data):
                    for c, cell_text in enumerate(row):
                        text_raw = str(cell_text) if cell_text is not None else ""
                        cells.append(TableCell(r=r, c=c, text_raw=text_raw))
                        
                        if r == 0:
                            col_headers.append(text_raw)
                        if c == 0:
                            row_headers.append(text_raw)
                
                tables.append(
                    Table(
                        table_id=tbl_id,
                        page=page_num,
                        bbox=tbl_bbox,
                        section_path="",
                        cells=cells,
                        row_headers=row_headers,
                        col_headers=col_headers,
                    )
                )
    
    document = Document(
        doc_id=doc_id,
        year=year,
        source_pdf=str(pdf_path),
        pages=pages,
        elements=elements,
        tables=tables,
    )
    
    report = {
        "doc_id": doc_id,
        "year": year,
        "source_pdf": str(pdf_path),
        "num_pages": len(pages),
        "num_elements": len(elements),
        "num_tables": len(tables),
        "failed_pages": [],
        "converter": "pdfplumber",
    }
    
    return document, report


def is_scanned_pdf(pdf_path: Path) -> bool:
    """Check if PDF is scanned (no text layer)."""
    with pdfplumber.open(pdf_path) as pdf:
        # Check first few pages for text
        for page in pdf.pages[:3]:
            text = page.extract_text()
            if text and len(text.strip()) > 50:
                return False
    return True


def generate_markdown(document: Document) -> str:
    """Generate human-readable markdown from document."""
    lines = []
    lines.append(f"# Document: {document.doc_id}")
    lines.append(f"Year: {document.year}")
    lines.append(f"Source: {document.source_pdf}")
    lines.append(f"Pages: {len(document.pages)}")
    lines.append("")
    
    for elem in document.elements:
        if elem.element_type == ElementType.HEADING and elem.text:
            lines.append(f"\n## {elem.text}")
            lines.append(f"*Page {elem.page}*\n")
        
        elif elem.element_type == ElementType.PARAGRAPH and elem.text:
            lines.append(f"{elem.text}\n")
    
    # Add table references
    if document.tables:
        lines.append("\n---\n## Tables\n")
        for table in document.tables:
            lines.append(f"- TABLE: {table.table_id} (page {table.page}, {len(table.cells)} cells)")
    
    return "\n".join(lines)


def save_tables_csv(document: Document, output_dir: Path) -> None:
    """Save tables as CSV files."""
    import csv
    
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    for table in document.tables:
        csv_path = tables_dir / f"{table.table_id}.csv"
        
        # Find max rows and cols
        if not table.cells:
            continue
            
        max_r = max(c.r for c in table.cells)
        max_c = max(c.c for c in table.cells)
        
        # Create grid
        grid: list[list[str]] = [[""] * (max_c + 1) for _ in range(max_r + 1)]
        for cell in table.cells:
            if cell.r <= max_r and cell.c <= max_c:
                grid[cell.r][cell.c] = cell.text_raw
        
        # Write CSV
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["table_id", "page", "bbox", "r", "c", "text_raw"])
            for r, row in enumerate(grid):
                for c, text in enumerate(row):
                    bbox_str = ",".join(str(v) for v in table.bbox.to_list())
                    writer.writerow([table.table_id, table.page, bbox_str, r, c, text])


def main(year: int) -> None:
    """Main ingest function."""
    raw_dir = get_raw_dir(year)
    output_dir = get_output_dir(year)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find PDF files
    pdf_files = list(raw_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {raw_dir}")
        return
    
    all_reports = []
    
    for pdf_path in pdf_files:
        print(f"Processing {pdf_path.name}...")
        
        try:
            # Check if PDF is scanned
            scanned = is_scanned_pdf(pdf_path)
            
            if scanned and OCR_AVAILABLE:
                print(f"  Detected scanned PDF, using OCR...")
                document, report = process_with_ocr(pdf_path, year)
            elif scanned and not OCR_AVAILABLE:
                print(f"  Warning: Scanned PDF detected but OCR not available")
                document, report = process_native_pdf(pdf_path, year)
            else:
                print(f"  Native PDF with text layer...")
                document, report = process_native_pdf(pdf_path, year)
            
            # Save outputs for each document
            # Use document name as suffix for multiple PDFs
            doc_name = pdf_path.stem.replace(" ", "_")
            
            # Save document.json
            write_json(document.model_dump(mode="json"), output_dir / f"document_{doc_name}.json")
            
            # Save document.md
            md_content = generate_markdown(document)
            (output_dir / f"document_{doc_name}.md").write_text(md_content, encoding="utf-8")
            
            # Save tables as CSV
            save_tables_csv(document, output_dir)
            
            all_reports.append(report)
            print(f"  OK: {report['num_pages']} pages, {report['num_elements']} elements, {report['num_tables']} tables")
        
        except Exception as e:
            print(f"  ERROR processing {pdf_path.name}: {e}")
            import traceback
            traceback.print_exc()
            all_reports.append(
                {
                    "source_pdf": str(pdf_path),
                    "error": str(e),
                    "converter": "failed",
                }
            )
    
    # Save ingest report
    write_json({"reports": all_reports}, output_dir / "ingest_report.json")
    print(f"\nIngest complete. Report saved to {output_dir / 'ingest_report.json'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline.00_ingest_docling YEAR")
        sys.exit(1)
    
    year = int(sys.argv[1])
    main(year)
