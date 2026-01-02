"""Number parsing and normalization utilities for Finnish financial data."""

import re
from typing import Optional, Tuple


def parse_finnish_number(text: str) -> Optional[float]:
    """Parse Finnish number format to float.
    
    Handles:
    - "1 234 567,89" → 1234567.89
    - "(123)" → -123
    - "−123" or "-123" → -123
    - "1 000 €" → 1000.0
    
    Returns:
        Parsed float or None if not parseable
    """
    if not text or not isinstance(text, str):
        return None
    
    text = text.strip()
    if not text:
        return None
    
    # Check for negative in parentheses: (123)
    is_negative = False
    if text.startswith("(") and text.endswith(")"):
        is_negative = True
        text = text[1:-1].strip()
    
    # Check for negative sign: − or -
    if text.startswith("−") or text.startswith("-"):
        is_negative = True
        text = text[1:].strip()
    
    # Remove currency symbols and other non-numeric chars except space, comma, dot
    text = re.sub(r"[^\d\s,\.]", "", text)
    
    # Replace Finnish decimal separator (comma) with dot
    # Handle both "1 234,56" and "1234,56"
    if "," in text:
        # Split by comma
        parts = text.split(",")
        if len(parts) == 2:
            integer_part = parts[0].replace(" ", "")
            decimal_part = parts[1]
            text = f"{integer_part}.{decimal_part}"
        else:
            # Multiple commas, treat as thousands separator
            text = text.replace(",", "")
    else:
        # Remove spaces (thousands separator)
        text = text.replace(" ", "")
    
    # Try to parse
    try:
        value = float(text)
        if is_negative:
            value = -value
        return value
    except ValueError:
        return None


def extract_unit_multiplier(text: str) -> Tuple[Optional[float], str]:
    """Extract unit multiplier from text.
    
    Handles:
    - "1 000 €" → (1000, "1 000 €")
    - "tuhatta euroa" → (1000, "tuhatta euroa")
    - "miljoonaa" → (1000000, "miljoonaa")
    
    Returns:
        Tuple of (multiplier, cleaned_text)
    """
    if not text:
        return None, text
    
    text_lower = text.lower()
    
    # Check for explicit multipliers in text
    multiplier_map = {
        "tuhatta": 1000,
        "tuhat": 1000,
        "miljoonaa": 1000000,
        "miljoona": 1000000,
        "miljardia": 1000000000,
        "miljardi": 1000000000,
    }
    
    for keyword, mult in multiplier_map.items():
        if keyword in text_lower:
            return mult, text
    
    # Check for "1 000 €" pattern
    thousand_pattern = r"1\s*000\s*€"
    if re.search(thousand_pattern, text):
        return 1000, text
    
    # Check for "1 000 000 €" pattern
    million_pattern = r"1\s*000\s*000\s*€"
    if re.search(million_pattern, text):
        return 1000000, text
    
    return None, text


def normalize_table_value(
    text_raw: str, unit_multiplier: Optional[float] = None
) -> Tuple[Optional[float], Optional[float]]:
    """Normalize table cell value.
    
    Args:
        text_raw: Raw cell text
        unit_multiplier: Optional unit multiplier to apply
        
    Returns:
        Tuple of (value_raw, value_eur) where value_eur is in euros
    """
    value_raw = parse_finnish_number(text_raw)
    
    if value_raw is None:
        return None, None
    
    if unit_multiplier:
        value_eur = value_raw * unit_multiplier
    else:
        value_eur = value_raw
    
    return value_raw, value_eur

