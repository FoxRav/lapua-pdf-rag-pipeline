"""Text cleaning and normalization utilities."""

import re
from typing import Optional


def clean_text(text: str) -> str:
    """Clean text: normalize whitespace, remove extra spaces."""
    if not text:
        return ""
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def normalize_heading(text: str) -> str:
    """Normalize heading text for section path matching."""
    text = clean_text(text)
    # Remove extra punctuation
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def extract_section_path(headings: list[dict[str, str]]) -> str:
    """Extract section path from heading hierarchy.
    
    Args:
        headings: List of heading dicts with 'level' and 'text' keys
        
    Returns:
        Section path like "H1 > H2 > H3"
    """
    if not headings:
        return ""
    
    # Filter and sort by level
    sorted_headings = sorted([h for h in headings if h.get("level")], key=lambda x: x["level"])
    if not sorted_headings:
        return ""
    
    return " > ".join(f"H{h['level']}" for h in sorted_headings)


def match_financial_statement_keywords(text: str) -> Optional[str]:
    """Match text against financial statement keywords.
    
    Returns:
        Statement type if matched, None otherwise.
    """
    text_lower = text.lower()
    
    keywords = {
        "income_statement": ["tuloslaskelma", "tulos", "tuloksen"],
        "balance_sheet": ["tase", "taseen"],
        "cash_flow": ["rahoituslaskelma", "rahoitus", "kassavirtalaskelma"],
        "investments": ["investoinnit", "investointi"],
        "contingent_liabilities": ["vastuut", "vastuu", "vastaavat"],
        "notes_index": ["liitetiedot", "liite", "selitykset"],
    }
    
    for statement_type, keyword_list in keywords.items():
        if any(keyword in text_lower for keyword in keyword_list):
            return statement_type
    
    return None

