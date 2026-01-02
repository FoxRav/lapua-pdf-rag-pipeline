"""I/O utilities for reading and writing pipeline data."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


def read_json(path: Path) -> dict[str, Any]:
    """Read JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(data: dict[str, Any] | list[Any], path: Path, indent: int = 2) -> None:
    """Write JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=indent, ensure_ascii=False), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL file."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").strip().split("\n") if line]


def write_jsonl(data: list[dict[str, Any]], path: Path) -> None:
    """Write JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_parquet(path: Path) -> pd.DataFrame:
    """Read Parquet file."""
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write Parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def get_data_dir() -> Path:
    """Get data directory from environment or default."""
    from dotenv import load_dotenv
    import os

    load_dotenv()
    data_dir = os.getenv("DATA_DIR", "./data")
    return Path(data_dir)


def get_output_dir(year: int) -> Path:
    """Get output directory for given year."""
    return get_data_dir() / "out" / str(year)


def get_raw_dir(year: int) -> Path:
    """Get raw data directory for given year."""
    return get_data_dir() / "raw" / str(year)

