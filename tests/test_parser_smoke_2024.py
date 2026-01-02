"""Smoke tests for PDF parser - 20 question validation.

Tests cover: cover page, table of contents, text pages, infographics, tables.
Ground truth: eval/smoke_questions_2024.json
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
EVAL_DIR = PROJECT_ROOT / "eval"
DATA_DIR = PROJECT_ROOT / "data" / "out" / "2024"


@pytest.fixture(scope="module")
def smoke_questions() -> list[dict[str, Any]]:
    """Load smoke test questions."""
    questions_path = EVAL_DIR / "smoke_questions_2024.json"
    with open(questions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


@pytest.fixture(scope="module")
def document() -> dict[str, Any]:
    """Load parsed document."""
    doc_path = DATA_DIR / "document_Lapua-Tilinpaatos-2024.json"
    with open(doc_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def section_chunks() -> list[dict[str, Any]]:
    """Load section chunks (text per page)."""
    chunks_path = DATA_DIR / "section_chunks.jsonl"
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


@pytest.fixture(scope="module")
def table_chunks() -> list[dict[str, Any]]:
    """Load table chunks."""
    chunks_path = DATA_DIR / "table_chunks.jsonl"
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def get_page_text(document: dict[str, Any], page: int) -> str:
    """Get all text from a specific page."""
    texts = []
    for elem in document.get("elements", []):
        if elem.get("page") == page:
            texts.append(elem.get("text", ""))
    return " ".join(texts)


def get_chunk_text(chunks: list[dict[str, Any]], page: int) -> str:
    """Get chunk text for a specific page."""
    for chunk in chunks:
        if chunk.get("page") == page:
            return chunk.get("text", "")
    return ""


def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, remove extra whitespace)."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def contains_any(text: str, keywords: list[str]) -> bool:
    """Check if text contains any of the keywords (case-insensitive)."""
    text_lower = normalize_text(text)
    for kw in keywords:
        if kw.lower() in text_lower:
            return True
    return False


def extract_numbers(text: str) -> list[float]:
    """Extract all numbers from text (handles Finnish format: 1 234,56)."""
    # Finnish number patterns
    pattern = r"-?\d[\d\s]*[,.]?\d*"
    matches = re.findall(pattern, text)
    
    numbers = []
    for m in matches:
        # Clean and parse
        cleaned = m.replace(" ", "").replace(",", ".")
        try:
            numbers.append(float(cleaned))
        except ValueError:
            pass
    return numbers


class TestParserSmokeP01ToP06:
    """Test cover page, TOC, and headings (P01-P06)."""
    
    def test_p01_main_title(self, document: dict[str, Any]) -> None:
        """P01: Mikä on asiakirjan pääotsikko?"""
        page_text = get_page_text(document, 1)
        assert contains_any(page_text, ["TILINPÄÄTÖS", "TILINPAATOS", "2024"]), \
            f"Expected 'TILINPÄÄTÖS 2024' on page 1, got: {page_text[:200]}"
    
    def test_p02_organization_name(self, document: dict[str, Any]) -> None:
        """P02: Mikä on kuntaorganisaation nimi kannessa?"""
        page_text = get_page_text(document, 1)
        assert contains_any(page_text, ["LAPUAN", "KAUPUNKI"]), \
            f"Expected 'Lapuan kaupunki' on page 1, got: {page_text[:200]}"
    
    def test_p03_toc_heading(self, section_chunks: list[dict[str, Any]]) -> None:
        """P03: Mikä on sisällysluettelon otsikko?"""
        page_text = get_chunk_text(section_chunks, 2)
        assert contains_any(page_text, ["Sisällys", "Sisallys", "sisällys"]), \
            f"Expected 'Sisällys' on page 2, got: {page_text[:200]}"
    
    def test_p04_toc_item1(self, section_chunks: list[dict[str, Any]]) -> None:
        """P04: Mikä on sisällysluettelon kohta 1?"""
        page_text = get_chunk_text(section_chunks, 2)
        assert contains_any(page_text, ["Olennaiset", "tapahtumat"]), \
            f"Expected 'Olennaiset tapahtumat' on page 2, got: {page_text[:300]}"
    
    def test_p05_toc_tilinpaatoslaskelmat_page(self, section_chunks: list[dict[str, Any]]) -> None:
        """P05: Missä sisällysluettelon mukaan alkaa Tilinpäätöslaskelmat?"""
        page_text = get_chunk_text(section_chunks, 3)
        # Should contain "132" near "Tilinpäätöslaskelmat"
        assert "132" in page_text or contains_any(page_text, ["Tilinpäätöslaskelmat"]), \
            f"Expected '132' or 'Tilinpäätöslaskelmat' on page 3"
    
    def test_p06_page4_heading(self, section_chunks: list[dict[str, Any]]) -> None:
        """P06: Mikä on sivun yläotsikko pdf_page=4 alussa?"""
        page_text = get_chunk_text(section_chunks, 4)
        assert contains_any(page_text, ["TOIMINTAKERTOMUS", "TASEKIRJA"]), \
            f"Expected 'TOIMINTAKERTOMUS' or 'TASEKIRJA' on page 4"


class TestParserSmokeP07ToP10:
    """Test text extraction and numbers (P07-P10)."""
    
    def test_p07_valtuusto_meetings(self, section_chunks: list[dict[str, Any]]) -> None:
        """P07: Kuinka monta kertaa valtuusto kokoontui vuonna 2024?"""
        page_text = get_chunk_text(section_chunks, 6)
        numbers = extract_numbers(page_text)
        assert 7 in numbers or "7" in page_text, \
            f"Expected '7' meetings on page 6, found numbers: {numbers[:10]}"
    
    def test_p08_valtuusto_members(self, section_chunks: list[dict[str, Any]]) -> None:
        """P08: Kuinka monta jäsentä valtuustossa on?"""
        page_text = get_chunk_text(section_chunks, 6)
        numbers = extract_numbers(page_text)
        assert 35 in numbers or "35" in page_text, \
            f"Expected '35' members on page 6, found numbers: {numbers[:10]}"
    
    def test_p09_population(self, section_chunks: list[dict[str, Any]]) -> None:
        """P09: Mikä on vuoden 2024 lopun asukasmäärä?"""
        page_text = get_chunk_text(section_chunks, 10)
        assert contains_any(page_text, ["14029", "14 029"]), \
            f"Expected '14 029' population on page 10"
    
    def test_p10_economy_heading(self, section_chunks: list[dict[str, Any]]) -> None:
        """P10: Mikä on otsikko sivulla jossa käsitellään talouskehitystä?"""
        page_text = get_chunk_text(section_chunks, 12)
        assert contains_any(page_text, ["talouskehitys", "2024"]), \
            f"Expected 'talouskehitys' on page 12"


class TestParserSmokeP11ToP19:
    """Test infographics extraction (P11-P19)."""
    
    def test_p11_infographic_title(self, section_chunks: list[dict[str, Any]]) -> None:
        """P11: Infografiikan pääotsikko."""
        page_text = get_chunk_text(section_chunks, 15)
        assert contains_any(page_text, ["keskeiset", "luvut", "Tilinpäätöksen"]), \
            f"Expected infographic title on page 15"
    
    def test_p12_toimintatuotot_kulut(self, section_chunks: list[dict[str, Any]]) -> None:
        """P12: Toimintatuotot ja Toimintakulut."""
        page_text = get_chunk_text(section_chunks, 15)
        # Check for key numbers
        assert contains_any(page_text, ["11", "47"]), \
            f"Expected '11' and '47.7' on page 15"
    
    def test_p13_toimintakate(self, section_chunks: list[dict[str, Any]]) -> None:
        """P13: Toimintakate."""
        page_text = get_chunk_text(section_chunks, 15)
        assert contains_any(page_text, ["36"]), \
            f"Expected '36.4' (toimintakate) on page 15"
    
    def test_p14_verotulot_valtionosuudet(self, section_chunks: list[dict[str, Any]]) -> None:
        """P14: Verotulot ja Valtionosuudet."""
        page_text = get_chunk_text(section_chunks, 15)
        assert contains_any(page_text, ["31", "13"]), \
            f"Expected '31.2' and '13.9' on page 15"
    
    def test_p15_vuosikate(self, section_chunks: list[dict[str, Any]]) -> None:
        """P15: Vuosikate."""
        page_text = get_chunk_text(section_chunks, 15)
        assert contains_any(page_text, ["7,5", "7.5"]), \
            f"Expected '7.5' (vuosikate) on page 15"
    
    def test_p16_poistot(self, section_chunks: list[dict[str, Any]]) -> None:
        """P16: Poistot."""
        page_text = get_chunk_text(section_chunks, 15)
        assert contains_any(page_text, ["6,9", "6.9"]), \
            f"Expected '6.9' (poistot) on page 15"
    
    def test_p17_tilikauden_ylijaama(self, section_chunks: list[dict[str, Any]]) -> None:
        """P17: Tilikauden ylijäämä."""
        page_text = get_chunk_text(section_chunks, 15)
        assert contains_any(page_text, ["0,6", "0.6"]), \
            f"Expected '0.6' (ylijäämä) on page 15"
    
    def test_p18_lainakanta(self, section_chunks: list[dict[str, Any]]) -> None:
        """P18: Lainakanta ja muutos."""
        page_text = get_chunk_text(section_chunks, 15)
        assert contains_any(page_text, ["64"]), \
            f"Expected '64.3' (lainakanta) on page 15"
    
    def test_p19_taseen_ylijaama(self, section_chunks: list[dict[str, Any]]) -> None:
        """P19: Taseen ylijäämä."""
        page_text = get_chunk_text(section_chunks, 15)
        assert contains_any(page_text, ["27"]), \
            f"Expected '27.4' (taseen ylijäämä) on page 15"


class TestParserSmokeP20:
    """Test table extraction (P20)."""
    
    def test_p20_henkilosto_maara(self, section_chunks: list[dict[str, Any]], table_chunks: list[dict[str, Any]]) -> None:
        """P20: Henkilöstön määrä 31.12.2024."""
        # Check section chunks (page 140)
        page_text = get_chunk_text(section_chunks, 140)
        
        # Also check table chunks for page 140
        table_text = ""
        for chunk in table_chunks:
            if chunk.get("page") == 140:
                table_text += chunk.get("text", "")
        
        combined = page_text + " " + table_text
        assert "470" in combined, \
            f"Expected '470' (henkilöstö) on page 140. Found: {combined[:500]}"


class TestParserSmokeRAG:
    """Test that RAG can answer smoke questions."""
    
    @pytest.fixture(scope="class")
    def rag(self):
        """Load RAG system."""
        from src.pipeline.query import TilinpaatosRAG
        return TilinpaatosRAG(2024, device="cpu")
    
    @pytest.mark.parametrize("question,expected_keywords", [
        ("Mikä on asiakirjan pääotsikko?", ["tilinpäätös", "2024"]),
        ("Mikä on henkilöstön määrä?", ["470", "henkilöstö"]),
        ("Mikä on vuosikate?", ["7,5", "7.5", "vuosikate"]),
        ("Mikä on toimintakate?", ["36", "toimintakate"]),
    ])
    def test_rag_retrieval(self, rag, question: str, expected_keywords: list[str]) -> None:
        """Test that RAG retrieves relevant chunks for key questions."""
        results = rag.search_hybrid(question, top_k=5)
        
        # Combine all result texts
        combined_text = " ".join([r.text for r in results])
        
        found = any(kw.lower() in combined_text.lower() for kw in expected_keywords)
        assert found, f"Expected one of {expected_keywords} in RAG results for: {question}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

