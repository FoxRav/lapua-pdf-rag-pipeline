"""Tests for number parsing utilities."""

import pytest

from src.common.num_parse import (
    extract_unit_multiplier,
    normalize_table_value,
    parse_finnish_number,
)


def test_parse_finnish_number_basic() -> None:
    """Test basic Finnish number parsing."""
    assert parse_finnish_number("1 234 567,89") == 1234567.89
    assert parse_finnish_number("1234,56") == 1234.56
    assert parse_finnish_number("1000") == 1000.0


def test_parse_finnish_number_negative() -> None:
    """Test negative number parsing."""
    assert parse_finnish_number("(123)") == -123.0
    assert parse_finnish_number("−123") == -123.0
    assert parse_finnish_number("-123") == -123.0
    assert parse_finnish_number("(1 234,56)") == -1234.56


def test_parse_finnish_number_with_currency() -> None:
    """Test parsing numbers with currency symbols."""
    assert parse_finnish_number("1 000 €") == 1000.0
    assert parse_finnish_number("123,45 €") == 123.45


def test_parse_finnish_number_invalid() -> None:
    """Test invalid number parsing."""
    assert parse_finnish_number("") is None
    assert parse_finnish_number("abc") is None
    assert parse_finnish_number("---") is None


def test_extract_unit_multiplier() -> None:
    """Test unit multiplier extraction."""
    mult, text = extract_unit_multiplier("1 000 €")
    assert mult == 1000
    
    mult, text = extract_unit_multiplier("tuhatta euroa")
    assert mult == 1000
    
    mult, text = extract_unit_multiplier("miljoonaa euroa")
    assert mult == 1000000
    
    mult, text = extract_unit_multiplier("123 €")
    assert mult is None


def test_normalize_table_value() -> None:
    """Test table value normalization."""
    value_raw, value_eur = normalize_table_value("1 234,56")
    assert value_raw == 1234.56
    assert value_eur == 1234.56
    
    value_raw, value_eur = normalize_table_value("1 000", unit_multiplier=1000)
    assert value_raw == 1000.0
    assert value_eur == 1000000.0
    
    value_raw, value_eur = normalize_table_value("(123)")
    assert value_raw == -123.0
    assert value_eur == -123.0

