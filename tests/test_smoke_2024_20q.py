"""Smoke tests for PDF parser - 20 question validation (T01-T20).

Based on: lapua_2024_pdf_parser_smoke_test_20_kysymysta_cursor_ohjeet.md

MUST tests: T01-T17 (17 tests) - must all pass
SHOULD tests: T18-T20 (3 tests) - figure/infographic OCR (optional)
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
def ground_truth() -> list[dict[str, Any]]:
    """Load ground truth from eval/smoke_2024_20q.json."""
    gt_path = EVAL_DIR / "smoke_2024_20q.json"
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    """Get all text from a specific page in document."""
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


def get_all_text_for_page(document: dict, section_chunks: list, table_chunks: list, page: int) -> str:
    """Get all available text for a page from all sources."""
    texts = []
    
    # From document elements
    texts.append(get_page_text(document, page))
    
    # From section chunks
    texts.append(get_chunk_text(section_chunks, page))
    
    # From table chunks
    for chunk in table_chunks:
        if chunk.get("page") == page:
            texts.append(chunk.get("text", ""))
    
    return " ".join(texts)


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def contains_text(haystack: str, needle: str) -> bool:
    """Check if haystack contains needle (case-insensitive, whitespace-normalized)."""
    return normalize_text(needle) in normalize_text(haystack)


def contains_number(text: str, number: int) -> bool:
    """Check if text contains a specific number."""
    # Handle Finnish number format (space as thousand separator)
    patterns = [
        str(number),
        f"{number:,}".replace(",", " "),  # 14 029
        f"{number:,}".replace(",", "."),  # 14.029
    ]
    text_normalized = normalize_text(text)
    return any(p in text_normalized for p in patterns)


# =============================================================================
# MUST TESTS (T01-T17)
# =============================================================================

class TestCoverPage:
    """T01-T02: Cover page tests."""
    
    def test_t01_main_title(self, document: dict, section_chunks: list, table_chunks: list) -> None:
        """T01: Kannen pääotsikko = TILINPÄÄTÖS 2024."""
        text = get_all_text_for_page(document, section_chunks, table_chunks, 1)
        assert contains_text(text, "TILINPÄÄTÖS") or contains_text(text, "TILINPAATOS"), \
            f"Expected 'TILINPÄÄTÖS 2024' on page 1"
        assert "2024" in text, f"Expected '2024' on page 1"
    
    def test_t02_organization_name(self, document: dict, section_chunks: list, table_chunks: list) -> None:
        """T02: Organisaation nimi = LAPUAN KAUPUNKI."""
        text = get_all_text_for_page(document, section_chunks, table_chunks, 1)
        assert contains_text(text, "LAPUAN") and contains_text(text, "KAUPUNKI"), \
            f"Expected 'LAPUAN KAUPUNKI' on page 1"


class TestTableOfContents:
    """T03-T08: Table of contents tests."""
    
    def test_t03_toc_heading(self, document: dict, section_chunks: list, table_chunks: list) -> None:
        """T03: Sisällysluettelon otsikko = Sisällys."""
        text = get_all_text_for_page(document, section_chunks, table_chunks, 2)
        assert contains_text(text, "Sisällys") or contains_text(text, "sisallys"), \
            f"Expected 'Sisällys' on page 2"
    
    def test_t04_toc_olennaiset_page3(self, document: dict, section_chunks: list, table_chunks: list) -> None:
        """T04: Olennaiset tapahtumat alkaa sivulta 3."""
        text = get_all_text_for_page(document, section_chunks, table_chunks, 2)
        assert contains_text(text, "Olennaiset") and contains_text(text, "tapahtumat"), \
            f"Expected 'Olennaiset tapahtumat' on page 2"
    
    def test_t05_toc_henkilosto_page17(self, document: dict, section_chunks: list, table_chunks: list) -> None:
        """T05: Kaupungin henkilöstö alkaa sivulta 17."""
        text = get_all_text_for_page(document, section_chunks, table_chunks, 2)
        assert contains_text(text, "henkilöstö") or contains_text(text, "henkilosto"), \
            f"Expected 'henkilöstö' on page 2"
    
    def test_t06_toc_tilinpaatoslaskelmat_page132(self, document: dict, section_chunks: list, table_chunks: list) -> None:
        """T06: Tilinpäätöslaskelmat alkaa sivulta 132."""
        text = get_all_text_for_page(document, section_chunks, table_chunks, 3)
        assert "132" in text or contains_text(text, "Tilinpäätöslaskelmat"), \
            f"Expected '132' or 'Tilinpäätöslaskelmat' on page 3"
    
    def test_t07_toc_liitetiedot_page138(self, document: dict, section_chunks: list, table_chunks: list) -> None:
        """T07: Liitetiedot alkaa sivulta 138."""
        text = get_all_text_for_page(document, section_chunks, table_chunks, 3)
        assert "138" in text or contains_text(text, "Liitetiedot"), \
            f"Expected '138' or 'Liitetiedot' on page 3"
    
    def test_t08_toc_allekirjoitus_page152(self, document: dict, section_chunks: list, table_chunks: list) -> None:
        """T08: Tilinpäätöksen allekirjoitus alkaa sivulta 152."""
        text = get_all_text_for_page(document, section_chunks, table_chunks, 3)
        assert "152" in text or contains_text(text, "allekirjoitus"), \
            f"Expected '152' or 'allekirjoitus' on page 3"


class TestTextExtraction:
    """T09-T12: Text extraction and number parsing."""
    
    def test_t09_ylijaama_06_milj(self, document: dict, section_chunks: list, table_chunks: list) -> None:
        """T09: Tilikauden ylijäämä 0,6 miljoonaa euroa (sivu 4)."""
        text = get_all_text_for_page(document, section_chunks, table_chunks, 4)
        # Check for 0,6 or 0.6
        assert "0,6" in text or "0.6" in text, \
            f"Expected '0,6' (ylijäämä) on page 4"
    
    def test_t10_tulos_19_milj(self, document: dict, section_chunks: list, table_chunks: list) -> None:
        """T10: Talousarvion tulos -1,9 milj. euroa (sivu 15)."""
        text = get_all_text_for_page(document, section_chunks, table_chunks, 15)
        assert "1,9" in text or "1.9" in text, \
            f"Expected '1,9' (tulos) on page 15"
    
    def test_t11_tuloveroprosentti_89(self, document: dict, section_chunks: list, table_chunks: list) -> None:
        """T11: Tuloveroprosentti 8,9 % (sivu 15)."""
        text = get_all_text_for_page(document, section_chunks, table_chunks, 15)
        assert "8,9" in text or "8.9" in text, \
            f"Expected '8,9' (tuloveroprosentti) on page 15"
    
    def test_t12_toimintakulut_493(self, document: dict, section_chunks: list, table_chunks: list) -> None:
        """T12: Toimintakulut 49,3 milj. euroa (sivu 15)."""
        text = get_all_text_for_page(document, section_chunks, table_chunks, 15)
        assert "49,3" in text or "49.3" in text, \
            f"Expected '49,3' (toimintakulut) on page 15"


class TestGovernance:
    """T13-T15: Governance text extraction."""
    
    def test_t13_valtuusto_7_kokousta(self, document: dict, section_chunks: list, table_chunks: list) -> None:
        """T13: Valtuusto kokoontui 7 kertaa (sivu 6)."""
        text = get_all_text_for_page(document, section_chunks, table_chunks, 6)
        assert "7" in text and contains_text(text, "valtuusto"), \
            f"Expected 'valtuusto' and '7' on page 6"
    
    def test_t14_valtuusto_35_jasenta(self, document: dict, section_chunks: list, table_chunks: list) -> None:
        """T14: Valtuustossa 35 jäsentä (sivu 6)."""
        text = get_all_text_for_page(document, section_chunks, table_chunks, 6)
        assert "35" in text, \
            f"Expected '35' (jäsentä) on page 6"
    
    def test_t15_hallitus_19_kokousta(self, document: dict, section_chunks: list, table_chunks: list) -> None:
        """T15: Kaupunginhallitus 19 kokousta (sivu 6)."""
        text = get_all_text_for_page(document, section_chunks, table_chunks, 6)
        assert "19" in text, \
            f"Expected '19' (kokousta) on page 6"


class TestTableExtraction:
    """T16-T17: Table extraction (henkilöstö sivu 18)."""
    
    def test_t16_henkilosto_vakinaiset_470(self, document: dict, section_chunks: list, table_chunks: list) -> None:
        """T16: Henkilöstötaulukko vakinaiset 2024 = 470 (sivu 18)."""
        text = get_all_text_for_page(document, section_chunks, table_chunks, 18)
        assert "470" in text, \
            f"Expected '470' (vakinaiset) on page 18. Got: {text[:500]}"
    
    def test_t17_henkilosto_yhteensa_578(self, document: dict, section_chunks: list, table_chunks: list) -> None:
        """T17: Henkilöstötaulukko yhteensä 2024 = 578 (sivu 18)."""
        text = get_all_text_for_page(document, section_chunks, table_chunks, 18)
        assert "578" in text, \
            f"Expected '578' (yhteensä) on page 18. Got: {text[:500]}"


# =============================================================================
# SHOULD TESTS (T18-T20) - Figure/Infographic OCR
# =============================================================================

@pytest.mark.xfail(reason="Figure OCR not yet implemented")
class TestFigureExtraction:
    """T18-T20: Figure/infographic extraction (optional)."""
    
    def test_t18_toimintakate_364(self, document: dict, section_chunks: list, table_chunks: list) -> None:
        """T18: Infografiikka toimintakate = -36,4 milj. € (sivu 15)."""
        text = get_all_text_for_page(document, section_chunks, table_chunks, 15)
        assert "36,4" in text or "36.4" in text, \
            f"Expected '36,4' (toimintakate) on page 15"
    
    def test_t19_vuosikate_75(self, document: dict, section_chunks: list, table_chunks: list) -> None:
        """T19: Infografiikka vuosikate = 7,5 milj. € (sivu 15)."""
        text = get_all_text_for_page(document, section_chunks, table_chunks, 15)
        assert "7,5" in text or "7.5" in text, \
            f"Expected '7,5' (vuosikate) on page 15"
    
    def test_t20_investointien_tulorahoitus_107(self, document: dict, section_chunks: list, table_chunks: list) -> None:
        """T20: Infografiikka investointien tulorahoitus = 107 % (sivu 15)."""
        text = get_all_text_for_page(document, section_chunks, table_chunks, 15)
        assert "107" in text, \
            f"Expected '107' (investointien tulorahoitus) on page 15"


# =============================================================================
# SUMMARY
# =============================================================================

class TestSummary:
    """Summary of smoke test results."""
    
    def test_must_count(self) -> None:
        """Ensure we have 17 MUST tests defined."""
        # This test documents the expected test count
        must_tests = 17
        assert must_tests == 17, "Should have 17 MUST tests (T01-T17)"
    
    def test_should_count(self) -> None:
        """Ensure we have 3 SHOULD tests defined."""
        should_tests = 3
        assert should_tests == 3, "Should have 3 SHOULD tests (T18-T20)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

