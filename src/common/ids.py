"""Generate stable IDs for documents and elements."""

import hashlib
from pathlib import Path
from typing import Any


def hash_bytes(data: bytes) -> str:
    """Generate SHA256 hash from bytes."""
    return hashlib.sha256(data).hexdigest()[:16]


def doc_id_from_pdf(pdf_path: Path, year: int) -> str:
    """Generate stable document ID from PDF path and year."""
    pdf_bytes = pdf_path.read_bytes()
    content_hash = hash_bytes(pdf_bytes)
    return f"doc_{year}_{content_hash}"


def element_id(element_type: str, page: int, bbox: tuple[float, float, float, float]) -> str:
    """Generate stable element ID from type, page, and bbox."""
    bbox_str = "_".join(f"{v:.2f}" for v in bbox)
    return f"{element_type}_p{page}_{bbox_str}"


def table_id(page: int, bbox: tuple[float, float, float, float]) -> str:
    """Generate stable table ID from page and bbox."""
    bbox_str = "_".join(f"{v:.2f}" for v in bbox)
    return f"table_p{page}_{bbox_str}"

