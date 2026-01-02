"""Create manifest.csv for all PDFs in a directory.

Creates manifest with: doc_id, filepath, sha256, pages_expected
"""

import csv
import hashlib
import sys
from pathlib import Path

# Try to use PyMuPDF for page count, fallback to None
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    print("Warning: PyMuPDF not installed, page counts will be None")


def sha256_file(filepath: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_page_count(filepath: Path) -> int | None:
    """Get page count of a PDF."""
    if not HAS_FITZ:
        return None
    try:
        doc = fitz.open(filepath)
        count = len(doc)
        doc.close()
        return count
    except Exception as e:
        print(f"Warning: Could not get page count for {filepath}: {e}")
        return None


def create_doc_id(filepath: Path) -> str:
    """Create a doc_id from filename."""
    # Remove extension and clean up
    name = filepath.stem
    # Replace special chars with underscore
    for char in [" ", "-", ".", "(", ")", "§", "–"]:
        name = name.replace(char, "_")
    # Remove multiple underscores
    while "__" in name:
        name = name.replace("__", "_")
    # Truncate if too long
    if len(name) > 50:
        name = name[:50]
    return name.lower().strip("_")


def create_manifest(pdf_dir: Path, output_path: Path) -> None:
    """Create manifest.csv for all PDFs in directory."""
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
    
    # Check for duplicates by sha256
    seen_hashes: dict[str, str] = {}
    rows = []
    
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}...")
        
        sha256 = sha256_file(pdf_path)
        
        # Check duplicate
        if sha256 in seen_hashes:
            print(f"  -> DUPLICATE of {seen_hashes[sha256]}, skipping")
            continue
        
        seen_hashes[sha256] = pdf_path.name
        
        doc_id = create_doc_id(pdf_path)
        pages = get_page_count(pdf_path)
        
        rows.append({
            "doc_id": doc_id,
            "filepath": str(pdf_path),
            "filename": pdf_path.name,
            "sha256": sha256,
            "pages_expected": pages or "",
        })
        
        print(f"  -> doc_id={doc_id}, pages={pages}")
    
    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "filepath", "filename", "sha256", "pages_expected"])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nManifest saved to: {output_path}")
    print(f"Total: {len(rows)} unique PDFs")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python create_manifest.py <pdf_directory> [output_path]")
        print("Example: python create_manifest.py data/raw_pdfs data/manifest.csv")
        sys.exit(1)
    
    pdf_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/manifest.csv")
    
    if not pdf_dir.exists():
        print(f"Error: Directory not found: {pdf_dir}")
        sys.exit(1)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    create_manifest(pdf_dir, output_path)


if __name__ == "__main__":
    main()

