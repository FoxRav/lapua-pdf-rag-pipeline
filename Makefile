.PHONY: ingest normalize extract chunk index eval all clean

YEAR ?= 2024
PYTHON := python
VENV := venv

# Aktivoi virtuaaliympäristö jos se on olemassa
ifeq ($(OS),Windows_NT)
	ACTIVATE := $(VENV)\Scripts\activate
else
	ACTIVATE := $(VENV)/bin/activate
endif

ingest:
	@echo "Ingesting $(YEAR)..."
	$(PYTHON) -m src.pipeline.00_ingest_docling $(YEAR)

normalize:
	@echo "Normalizing $(YEAR)..."
	$(PYTHON) -m src.pipeline.01_normalize $(YEAR)

extract:
	@echo "Extracting schema $(YEAR)..."
	$(PYTHON) -m src.pipeline.02_extract_schema $(YEAR)

chunk:
	@echo "Chunking $(YEAR)..."
	$(PYTHON) -m src.pipeline.03_chunk $(YEAR)

index:
	@echo "Indexing $(YEAR)..."
	$(PYTHON) -m src.pipeline.04_index $(YEAR)

eval:
	@echo "Evaluating $(YEAR)..."
	$(PYTHON) -m src.pipeline.05_eval $(YEAR)

all: ingest normalize extract chunk index eval
	@echo "Pipeline complete for $(YEAR)"

clean:
	@echo "Cleaning interim and out directories..."
	rm -rf data/interim/*
	rm -rf data/out/*

