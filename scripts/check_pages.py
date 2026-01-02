"""Check what data exists on specific pages."""
import json
from pathlib import Path

DATA_DIR = Path("data/out/2024")

# Check section chunks for pages 130+
print("=== Section chunks (pages 130+) ===")
with open(DATA_DIR / "section_chunks.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            c = json.loads(line)
            if c.get("page", 0) >= 130:
                text = c.get("text", "")[:150].replace("\n", " ")
                print(f"Page {c['page']}: {text}...")

print("\n=== Table chunks (pages 130+) ===")
with open(DATA_DIR / "table_chunks.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            c = json.loads(line)
            if c.get("page", 0) >= 130:
                text = c.get("text", "")[:150].replace("\n", " ")
                print(f"Page {c['page']} [{c.get('table_id','')}]: {text}...")

