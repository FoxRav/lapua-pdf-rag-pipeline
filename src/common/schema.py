"""Financial statement schema definitions."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel


class ElementType(str, Enum):
    """Element type enumeration."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    CAPTION = "caption"
    LIST = "list"
    FIGURE = "figure"


class StatementType(str, Enum):
    """Financial statement type enumeration."""

    INCOME_STATEMENT = "income_statement"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"
    INVESTMENTS = "investments"
    CONTINGENT_LIABILITIES = "contingent_liabilities"
    NOTES_INDEX = "notes_index"


class BBox(BaseModel):
    """Bounding box coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float

    def to_list(self) -> list[float]:
        """Convert to list format [x0, y0, x1, y1]."""
        return [self.x0, self.y0, self.x1, self.y1]

    @classmethod
    def from_list(cls, coords: list[float]) -> "BBox":
        """Create from list format [x0, y0, x1, y1]."""
        if len(coords) != 4:
            raise ValueError(f"BBox requires 4 coordinates, got {len(coords)}")
        return cls(x0=coords[0], y0=coords[1], x1=coords[2], y1=coords[3])


class Element(BaseModel):
    """Document element (text, heading, table, etc.)."""

    element_id: str
    element_type: ElementType
    page: int
    bbox: BBox
    section_path: str
    text: str | None = None


class TableCell(BaseModel):
    """Table cell."""

    r: int  # row index
    c: int  # column index
    text_raw: str


class Table(BaseModel):
    """Table element."""

    table_id: str
    page: int
    bbox: BBox
    section_path: str
    cells: list[TableCell]
    row_headers: list[str] = []
    col_headers: list[str] = []


class Page(BaseModel):
    """Document page."""

    page_num: int
    width: float
    height: float


class Document(BaseModel):
    """Canonical document structure."""

    doc_id: str
    year: int
    source_pdf: str
    pages: list[Page]
    elements: list[Element]
    tables: list[Table] = []


class LineItem(BaseModel):
    """Financial statement line item."""

    year: int
    doc_id: str
    statement: StatementType
    line_item_key: str
    label_original: str
    value_eur: float
    source_table_id: str
    page: int
    bbox: BBox

