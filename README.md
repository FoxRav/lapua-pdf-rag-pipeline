# 🏛️ Lapua PDF RAG Pipeline

### PDF-dokumentit → Strukturoitu data → 1024-dimensioinen vektoriavaruus → Älykäs Q&A

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![GPU Accelerated](https://img.shields.io/badge/GPU-CUDA%2012.4-brightgreen.svg)](#gpu-tuki)
[![Embedding: BGE-M3](https://img.shields.io/badge/Embedding-BGE--M3-orange.svg)](#arkkitehtuuri)

---

## 🎯 Mitä tämä tekee?

**Syötä sisään PDF-tiedostoja — saat ulos AI-valmiin tietokannan.**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  📄 PDF:t       │ ─▶ │  🔧 PARSE       │ ─▶ │  🧮 EMBED       │ ─▶ │  🎯 QUERY       │
│  1...N kpl      │    │  Teksti+Taulut  │    │  1024-dim       │    │  Semanttinen    │
│  Tilinpäätökset │    │  Strukturoitu   │    │  Vektori-       │    │  haku + LLM     │
│  Talousarviot   │    │  JSON/CSV       │    │  avaruus        │    │  vastaus        │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Pipeline tuottaa:

| Output | Formaatti | Käyttötarkoitus |
|--------|-----------|-----------------|
| **Strukturoitu teksti** | JSONL | Sivu, kappale, bbox, metadata |
| **Taulukkodata** | CSV + JSON | Rivit, sarakkeet, solut — laskettavissa |
| **Vektori-indeksi** | FAISS (1024-dim) | Semanttinen samankaltaisuushaku |
| **BM25-indeksi** | Pickle | Avainsanapohjainen haku |
| **Chunk-metadata** | JSON | Jäljitettävyys: sivu, taulukko, lähde |

### Yksi dokumentti vai tuhat?

```bash
# Yksi PDF
python -m src.pipeline.batch_ingest manifest.csv --limit 1

# Kaikki 25 PDF:ää
python -m src.pipeline.batch_ingest manifest.csv

# Skaalautuu: 2000 PDF:ää samalla tavalla
```

**Lopputulos:** Tekoälyvalmiiksi prosessoitu dokumenttikokoelma, josta voit:
- 🔍 Hakea semanttisesti ("Mikä oli vuosikate?")
- 📊 Ajaa analytiikkaa (taulukot CSV:nä)
- 🤖 Generoida vastauksia LLM:llä (RAG)
- ✅ Validoida parserin laatu (50+ smoke-testiä)

---

## 💡 Miksi tämä?

| Ongelma | Ratkaisu |
|---------|----------|
| 📄 154-sivuinen PDF | ⚡ Vastaus 3 sekunnissa |
| 🔍 Etsi Ctrl+F | 🧠 Kysy luonnollisella kielellä |
| 📊 Taulukot kuvina | 📈 Strukturoitu, laskettava data |
| 🤷 "Missä tämä luku on?" | 📍 Sivunumero + tarkka lähde |
| 🗂️ 25 dokumenttia | 🚀 Yksi yhtenäinen vektori-indeksi |

**Esimerkki:**
```
Kysymys: "Paljonko oli poistoja vuonna 2024?"
Vastaus: "Poistot olivat 6 832 049 euroa. (sivu 140, tuloslaskelma)"
```

---

## 🚀 PIKAOHJE: Näin käytät

### 1. Aktivoi ympäristö (aina ensin!)

```powershell
cd F:\-DEV-\33.Lapua-tilinpäätös2025-20250201
.\venv_gpu\Scripts\Activate.ps1
$env:PYTHONPATH = "."
```

### 2. Kysy kysymys tilinpäätöksestä

```powershell
# Yksinkertainen haku (näyttää löydetyt tekstit)
python -m src.pipeline.query 2024 "Mikä on vuosikate?"

# Täysi vastaus tekoälyltä (käyttää Lapua-LLM:ää)
python -m src.pipeline.rag_answer 2024 "Mikä on vuosikate euroina?"
```

### 3. Esimerkkejä kysymyksistä

```powershell
# Talouden tunnusluvut
python -m src.pipeline.rag_answer 2024 "Paljonko on poistoja?"
python -m src.pipeline.rag_answer 2024 "Mikä on lainakanta?"
python -m src.pipeline.rag_answer 2024 "Mikä on tilikauden ylijäämä?"

# Henkilöstö
python -m src.pipeline.rag_answer 2024 "Kuinka paljon on henkilöstöä?"

# Toiminta
python -m src.pipeline.rag_answer 2024 "Kuinka monta kurssia oli palvelukodeissa?"
```

---

## 📊 Miten tämä toimii?

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  KYSYMYS     │ ──▶ │  HAKU        │ ──▶ │  KONTEKSTI   │ ──▶ │  VASTAUS     │
│  "Mikä on    │     │  Etsii PDF:n │     │  Top-5       │     │  Tekoäly     │
│  vuosikate?" │     │  tekstistä   │     │  osumaa      │     │  vastaa      │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

**Vaihe 1:** Kirjoitat kysymyksen suomeksi  
**Vaihe 2:** Järjestelmä etsii PDF:stä 5 parasta osumaa (BM25 + vektorihaku)  
**Vaihe 3:** Tekoäly (Lapua-LLM) lukee löydetyt tekstit ja vastaa kysymykseen  
**Vaihe 4:** Saat vastauksen + lähteet (sivunumero, taulukko)

---

## 📁 Mitä dataa järjestelmässä on?

| Dokumentti | Sivuja | Taulukoita | Tekstiä |
|------------|--------|------------|---------|
| Lapua-Tilinpaatos-2024.pdf | 154 | 123 | 269 sivua |
| Lapuan-kaupunki-Talousarvio-2025.pdf | 117 | 76 | 116 sivua |

**Yhteensä 478 hakukelpoista tekstipalaa (chunk).**

---

## 🧪 Testaa että kaikki toimii

```powershell
# Aja uudet smoke testit (20 kysymystä spesifikaation mukaan)
python -m pytest tests/test_smoke_2024_20q.py -v

# Vanha testiajo (vielä toimii)
python -m pytest tests/test_parser_smoke_2024.py -v
```

### Testitulokset (2025-01-02)

| Tyyppi | Testit | Tulos |
|--------|--------|-------|
| **MUST** (T01-T17) | Kansi, TOC, teksti, numerot, taulukot | 17/17 ✅ |
| **SHOULD** (T18-T20) | Infografiikka (figure OCR) | 3/3 ✅ |

Jos näet `19 passed, 3 xpassed` → kaikki toimii! ✅

### Smoke test -kysymykset (T01-T20)

**MUST (17 kpl):**
- T01-T02: Kansi (pääotsikko, organisaatio)
- T03-T08: Sisällysluettelo (sivunumerot)
- T09-T12: Tekstin numerot (ylijäämä, tuloveroprosentti, toimintakulut)
- T13-T15: Hallinto (valtuusto, kaupunginhallitus)
- T16-T17: Henkilöstötaulukko (vakinaiset 470, yhteensä 578)

**SHOULD (3 kpl):**
- T18-T20: Infografiikka sivu 15 (toimintakate, vuosikate, tulorahoitus)

---

## ⚙️ Tekninen tausta

### Käytetyt mallit

| Malli | Tarkoitus | Koko |
|-------|-----------|------|
| `BAAI/bge-m3` | Tekstin vektorisointi (haku) | 568M |
| `Qwen/Qwen2.5-1.5B-Instruct` | Tekoälyn pohjamalli | 1.5B |
| `CCG-FAKTUM/lapua-llm-v2` | LoRA-adapteri (Lapuan kieli) | 10M |

### Periaatteet

- Tekoäly EI laske lukuja itse → vain viittaa PDF:n tekstiin
- Kaikilla luvuilla on lähde: sivu + taulukko
- Sama järjestelmä toimii muillekin tilinpäätöksille

---

## 💡 Vastauksen tulkinta

Kun ajat kyselyn, saat vastauksen tässä muodossa:

```
============================================================
KYSYMYS: Paljonko on poistoja?
============================================================

VASTAUS (CCG-FAKTUM/lapua-llm-v2):

Johtopäätös: Vuonna 2024 poistoja oli 6,8 miljoonaa euroa.

Perustelut: Suunnitelman mukaiset poistot olivat -6 832 049,39 euroa.
Lisäksi arvonalentumiset olivat -34 080,94 euroa.

Lähteet: Sivu 128, taulukko.

------------------------------------------------------------
LÄHTEET:
  1. Sivu 128 | table_p128_...
  2. Sivu 133 | table_p133_...
  3. Sivu 142
```

**Huomaa:**
- **Johtopäätös** = Suora vastaus kysymykseen
- **Perustelut** = Miten vastaus on johdettu PDF:stä
- **Lähteet** = Sivunumerot joista tieto löytyy

---

## Nykytilanne (2025-01-02)

### Toimiva GPU-tuettu pipeline

| Vaihe | Moduuli | GPU-käyttö | Tila |
|-------|---------|------------|------|
| 00_ingest | `00_ingest_docling.py` | OCR: CPU (RapidOCR) | ✅ Toimii |
| 01_normalize | `01_normalize.py` | - | ✅ Toimii |
| 02_extract | `02_extract_schema.py` | - | ✅ Toimii |
| 03_chunk | `03_chunk.py` | - | ✅ Toimii |
| 04_index | `04_index.py` | **GPU (CUDA)** embeddings | ✅ Toimii |
| 05_eval | `05_eval.py` | - | ✅ Toimii |

### Prosessoidut dokumentit (2024)

| Dokumentti | Tyyppi | Sivut | Elementit | Taulukot | Lähde |
|-----------|--------|-------|-----------|----------|-------|
| Lapua-Tilinpaatos-2024.pdf | Skannattu (OCR) | 154 | 610 | **123** | PDF_Parser (PP-StructureV3) |
| Lapuan-kaupunki-Talousarvio-2025.pdf | Natiivi PDF | 117 | 116 | 74 | pdfplumber |

### PDF_Parser-integraatio (uusi)

Koska skannatut PDF:t ovat haastavia, käytetään `F:\-DEV-\PDF_Parser`-projektin PP-StructureV3 (PaddleOCR) parseria:

```bash
# Tuo taulukot PDF_Parser-projektista
python -m src.pipeline.import_pdfparser 2024
```

**Tulokset:**
- 123 taulukkoa tunnistettu ja tuotu
- Sisältää tuloslaskelma, rahoituslaskelma, tase, konsernitilinpäätös
- 209 low-confidence solua merkitty (lähinnä viivaelementtejä)
- Ei kirjanpidon validointivirheitä

### Pipeline-tulokset (päivitetty 2025-01-02)

- **Normalize**: 11015 taulukkosolua, 61.5% numeerisia
- **Extract**: 509 line itemiä (17 tuloslaskelmaa, 7 tasetta, 1 rahoituslaskelma, 2 investointia)
- **Chunk**: 478 chunkkia (269 teksti + 199 taulukko + 10 statement)
- **Index**: Hybridi-indeksi (BM25 + FAISS BGE-M3), GPU-embeddings (~61s)
- **RAG**: Toimiva kysely + LLM-vastausten generointi (Lapua-LLM LoRA)

### Kuvakansiot

```
data/out/2024/
├── page_images/      # 154 sivukuvaa (PDF_Parserista)
├── table_grids/      # 212 taulukkogridia (PP-StructureV3)
├── tables/           # 181 CSV-taulukkoa
└── tilinpaatos_tables/  # 147 tilinpäätöstaulukkoa
```

---

## Asennus

### Vaatimukset

- Python 3.10+
- CUDA 12.4+ (GPU-tuki valinnainen mutta suositeltava)
- Windows 10/11 tai Linux

### Perusasennus (CPU)

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -e .
```

### GPU-asennus (suositeltu)

```bash
# Luo erillinen GPU-ympäristö
python -m venv venv_gpu
venv_gpu\Scripts\activate  # Windows

# PyTorch CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# ONNX Runtime GPU
pip install onnxruntime-gpu

# Loput riippuvuudet
pip install -e .
```

### GPU-tuen testaus

```bash
# Aktivoi GPU-ympäristö ensin!
.\venv_gpu\Scripts\Activate.ps1

# Testaa PyTorch CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# Testaa ONNX Runtime
python -c "import onnxruntime as ort; print('Providers:', ort.get_available_providers())"

# Testaa Sentence Transformers GPU (BGE-M3)
python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('BAAI/bge-m3', device='cuda'); print('BGE-M3 GPU OK')"
```

---

## Käyttö

### Ennen ajoa

1. Aktivoi oikea virtuaaliympäristö:
   ```bash
   .\venv_gpu\Scripts\Activate.ps1  # GPU-tuki
   # tai
   .\venv\Scripts\Activate.ps1      # CPU-only
   ```

2. Kopioi PDF:t oikeaan kansioon:
   ```
   data/raw/{YEAR}/
   ```

### Pipeline-komennot

```bash
# Koko putki yhdellä komennolla
make all YEAR=2024

# Tai vaiheittain (suositeltava debuggaukseen)
make ingest YEAR=2024     # PDF → JSON + MD + CSV-taulukot
make normalize YEAR=2024  # Normalisoi luvut ja yksiköt
make extract YEAR=2024    # Poimi tilinpäätösrivit skeemaan
make chunk YEAR=2024      # Luo RAG-chunkit
make index YEAR=2024      # Rakenna BM25 + vektori-indeksi (GPU)
make eval YEAR=2024       # Evaluoi ja tarkista data
```

### Suora Python-käyttö

```bash
python -m src.pipeline.00_ingest_docling 2024
python -m src.pipeline.01_normalize 2024
python -m src.pipeline.02_extract_schema 2024
python -m src.pipeline.03_chunk 2024
python -m src.pipeline.04_index 2024
python -m src.pipeline.05_eval 2024
```

### RAG-haku tilinpäätöstiedoista

```bash
# Yksittäinen kysymys
python -m src.pipeline.query 2024 "Mikä on vuosikate?"
python -m src.pipeline.query 2024 "Paljonko on lainakanta?"
python -m src.pipeline.query 2024 "tuloslaskelma toimintakate"

# Interaktiivinen tila
python -m src.pipeline.query 2024
```

**Esimerkkitulos:**
```
--- Tulos 1 (score: 0.787) [Sivu 28] [Taulukko: table_p28...] ---
TULOSLASKELMAN TUNNUSLUVUT | 2024 | 2023
Vuosikate/poistot, % | 109,3% | 167,7%
Vuosikate €/asukas | 535€ | 794€

--- Tulos 2 (score: 0.762) [Sivu 32] [Taulukko: table_p32...] ---
Vuosikate | 7502 | 11140
```

RAG käyttää hybridi-hakua (BM25 + vektori, GPU-embeddings).

### RAG + Lapua-LLM (vastausten generointi)

```bash
# Käytä Lapua-LLM LoRA-adapteria vastausten generointiin
python -m src.pipeline.rag_answer 2024 "Paljonko on vuosikate euroina?"

# Interaktiivinen tila
python -m src.pipeline.rag_answer 2024
```

**LoRA-adapteri:** [CCG-FAKTUM/lapua-llm-v2](https://huggingface.co/CCG-FAKTUM/lapua-llm-v2)
- Pohjamalli: Qwen/Qwen2.5-1.5B-Instruct
- Hienosäädetty Lapuan kaupungin hallintoteksteille
- Vastausformaatti: Johtopäätös → Perustelut → Lähteet

---

## RAG-arkkitehtuuri (Retrieval-Augmented Generation)

### Putken yleiskuva

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
│  KYSYMYS    │───▶│  EMBEDDING   │───▶│   HAKU      │───▶│  LLM        │───▶ VASTAUS
│  (teksti)   │    │  (BGE-M3)    │    │  (Hybridi)  │    │  (Qwen+LoRA)│
└─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘
```

### Vaihe 1: Embedding (Vektorisointi)

**Malli:** `BAAI/bge-m3` (sentence-transformers)

| Ominaisuus | Arvo |
|------------|------|
| Parametrit | ~568M |
| Vektoriulottuvuus | 1024 |
| Max tokens | 8192 |
| Kielituki | 100+ kieltä (suomi ✅) |
| Laite | GPU (CUDA) |

Kysymys muunnetaan 1024-ulotteiseksi vektoriksi semanttista hakua varten.

### Vaihe 2: Hybridi-haku (BM25 + Vektori)

Käytetään kahta hakumenetelmää rinnakkain:

#### A) BM25 Sparse Search (50% painotus)
- **Algoritmi:** BM25Okapi (rank_bm25-kirjasto)
- **Toiminta:** Sanojen esiintymistiheys ja harvinaisuus
- **Vahvuus:** Tarkat sanahaut ("poistot", "vuosikate", "euroa")
- **Laite:** CPU

#### B) Vektori-haku (50% painotus)
- **Indeksi:** FAISS IndexFlatL2
- **Toiminta:** Kosini-samankaltaisuus vektoriavaruudessa
- **Vahvuus:** Semanttinen ymmärrys, synonyymit
- **Laite:** CPU (indeksi pieni, ~478×1024)

#### C) Yhdistäminen
```python
hybrid_score = 0.5 * bm25_score + 0.5 * vector_score
top_chunks = sorted(all_chunks, by=hybrid_score)[:5]
```

### Vaihe 3: Kontekstin muodostus

Top-5 parasta chunkkia yhdistetään kontekstitekstiksi (max 6000 merkkiä):

```
[Sivu 95] Lapuan kaupunki Tilinpäätös 2024...
[Sivu 128] Suunnitelman mukaiset poistot -6 832 049,39...
[Sivu 141] Poistosuunnitelma: Rakennukset 25-50 vuotta...
```

### Vaihe 4: LLM-generointi

#### Pohjamalli
**Malli:** `Qwen/Qwen2.5-1.5B-Instruct`

| Ominaisuus | Arvo |
|------------|------|
| Parametrit | 1.5B |
| Kvantisaatio | 4-bit (BitsAndBytes) |
| GPU-muisti | ~2GB |
| Laite | GPU (CUDA) |

#### LoRA-adapteri
**Malli:** `CCG-FAKTUM/lapua-llm-v2`

| Ominaisuus | Arvo |
|------------|------|
| Parametrit | ~10M (adapteri) |
| Tarkoitus | Lapuan hallintokielen hienosäätö |
| Vastausformaatti | Johtopäätös → Perustelut → Lähteet |

#### Prompt-rakenne
```
System: Olet Lapuan kaupungin tilinpäätösasiantuntija. 
LUE KONTEKSTI HUOLELLISESTI ja etsi sieltä TARKAT NUMEROT...

User: Konteksti (tilinpäätöstiedot 2024):
[Sivu 128] Suunnitelman mukaiset poistot -6 832 049,39...

Kysymys: Paljonko on poistoja ja mitä ne ovat?
```

#### Generointiparametrit
```python
outputs = model.generate(
    inputs,
    max_new_tokens=512,
    temperature=0.3,
    do_sample=True
)
```

### Mallit yhteenvetona

| Vaihe | Malli | Koko | Laite |
|-------|-------|------|-------|
| Embedding | `BAAI/bge-m3` | 568M | GPU (CUDA) |
| Sparse-haku | BM25Okapi | - | CPU |
| Vektori-indeksi | FAISS IndexFlatL2 | 478×1024 | CPU |
| LLM (pohja) | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | GPU (4-bit) |
| LLM (LoRA) | `CCG-FAKTUM/lapua-llm-v2` | ~10M | GPU |

### Suoritusajat

| Vaihe | Aika |
|-------|------|
| BGE-M3 lataus | ~30s (kerran) |
| Qwen + LoRA lataus | ~20s (kerran) |
| Kysely-embedding | ~0.1s |
| Hybridi-haku | ~0.05s |
| LLM-generointi | ~5-10s |
| **Ensimmäinen kysely** | **~60s** (sis. mallien lataus) |
| **Seuraavat kyselyt** | **~10s** |

### Indeksin sisältö (2024)

| Chunk-tyyppi | Määrä | Kuvaus |
|--------------|-------|--------|
| Teksti (sivut) | 269 | Sivukohtaiset tekstikappaleet |
| Taulukot | 199 | Markdown-muotoiset taulukot |
| Tilinpäätösrivit | 10 | Ryhmitellyt line itemit |
| **Yhteensä** | **478** | |

---

## Pipeline-vaiheet

### 00_ingest_docling.py

**Syöte**: `data/raw/{YEAR}/*.pdf`  
**Tuloste**: `data/out/{YEAR}/document_*.json`, `*.md`, `tables/*.csv`

- Lukee PDF:t käyttäen pdfplumber + RapidOCR (skannatuille)
- Tunnistaa otsikot, kappaleet, taulukot
- Luo kanoninen Document-malli (JSON)
- Generoi markdown-version
- Tallentaa taulukot CSV:nä

**Huom**: Skannatuista PDF:istä ei tunnisteta taulukoita automaattisesti (vain teksti OCR:llä).

### 01_normalize.py

**Syöte**: `document_*.json`  
**Tuloste**: `normalized_tables.parquet`, `normalized_text.jsonl`

- Normalisoi suomalaiset numerot (1 234,56 → 1234.56)
- Tunnistaa yksiköt (1000€, milj.€)
- Käsittelee negatiiviset luvut (suluissa, miinusmerkki)

### 02_extract_schema.py

**Syöte**: `normalized_tables.parquet`, `document_*.json`  
**Tuloste**: `financial_{YEAR}.json`, `line_items_long.csv`

- Tunnistaa tilinpäätöstaulukot (tuloslaskelma, tase, kassavirta)
- Poimii rivit kanoniseen skeemaan
- Tukee schema_map.yaml-konfiguraatiota

### 03_chunk.py

**Syöte**: `document_*.json`, `line_items_long.csv`  
**Tuloste**: `section_chunks.jsonl`, `table_chunks.jsonl`, `statement_chunks.jsonl`

- Luo RAG-chunkit dokumenteista
- Section chunks: otsikko + seuraavat kappaleet
- Table chunks: taulukko markdown-muodossa
- Statement chunks: tilinpäätösrivit ryhmiteltynä

### 04_index.py (GPU)

**Syöte**: `*_chunks.jsonl`  
**Tuloste**: `index/bm25.pkl`, `index/faiss.index`, `index/metadata.json`

- Rakentaa BM25-indeksin (sparse retrieval)
- Luo vektoriembeddinkit (sentence-transformers, GPU)
- Tallentaa FAISS-indeksin (dense retrieval)
- Käyttää `intfloat/multilingual-e5-large` -mallia

### 05_eval.py

**Syöte**: `index/`, `line_items_long.csv`  
**Tuloste**: `reconcile_report.json`, `retrieval_eval.json`, `eval_report.json`

- Tarkistaa datan eheys (reconciliation)
- Evaluoi retrieval-laatua (tulossa)

---

## Repo-rakenne

```
finstmt-rag/
├── data/
│   ├── raw/                    # Alkuperäiset PDF:t
│   │   └── {YEAR}/             # Esim. 2024/
│   ├── interim/                # Väliaikaiset tiedostot
│   └── out/                    # Prosessoidut tiedostot
│       └── {YEAR}/
│           ├── document_*.json # Kanoninen dokumenttimalli
│           ├── document_*.md   # Markdown-versio
│           ├── tables/         # Taulukot CSV:nä
│           ├── normalized_*.   # Normalisoidut datat
│           ├── financial_*.json# Tilinpäätösskeema
│           ├── *_chunks.jsonl  # RAG-chunkit
│           └── index/          # BM25 + FAISS indeksit
├── src/
│   ├── common/                 # Yhteiset moduulit
│   │   ├── ids.py              # Stabiilit ID:t (hash)
│   │   ├── io.py               # JSON/JSONL/Parquet I/O
│   │   ├── num_parse.py        # Suomalaisten lukujen parsinta
│   │   ├── schema.py           # Pydantic-mallit (Document, Element, Table)
│   │   └── text_clean.py       # Tekstin siivous
│   └── pipeline/               # ETL-vaiheet (00-05)
├── configs/
│   ├── pipeline.yaml           # Pipeline-asetukset
│   ├── schema_map.yaml         # Tilinpäätösrivien mappaus
│   └── stopwords_fi.txt        # Suomalaiset stopwordit
├── tests/                      # Pytest-testit
├── venv/                       # CPU-virtuaaliympäristö
├── venv_gpu/                   # GPU-virtuaaliympäristö (CUDA)
├── Makefile                    # Pipeline-komennot
├── pyproject.toml              # Python-projektin konfiguraatio
└── README.md                   # Tämä tiedosto
```

---

## Tietomalli (schema.py)

### Document

```python
class Document:
    doc_id: str              # Uniikki hash
    year: int                # Tilinpäätösvuosi
    source_pdf: str          # Alkuperäinen PDF
    pages: list[Page]        # Sivut
    tables: list[Table]      # Taulukot
```

### Element

```python
class Element:
    element_id: str
    element_type: Literal["heading", "paragraph", "table", "list_item"]
    text: str
    page: int
    bbox: BBox               # (x0, y0, x1, y1)
    heading_level: int | None
    section_path: list[str]  # Otsikkopolku
```

### Table

```python
class Table:
    table_id: str
    page: int
    bbox: BBox
    cells: list[TableCell]
    num_rows: int
    num_cols: int
```

### LineItem (tilinpäätösrivi)

```python
class LineItem:
    year: int
    doc_id: str
    statement: StatementType  # income_statement, balance_sheet, ...
    label: str               # Alkuperäinen rivi
    canonical_label: str     # Normalisoitu nimi
    value_eur: float | None
    page: int
    bbox: BBox
```

---

## Konfiguraatio

### configs/pipeline.yaml

```yaml
embedding_model: "BAAI/bge-m3"
embedding_device: "cuda"  # tai "cpu"
chunk_max_tokens: 8192  # BGE-M3 tukee pitkiä konteksteja
bm25_weight: 0.5
vector_weight: 0.5
llm_model: "Qwen/Qwen2.5-1.5B-Instruct"
lora_adapter: "CCG-FAKTUM/lapua-llm-v2"
```

### configs/schema_map.yaml

Mappaa tilinpäätösrivien nimet kanonisiin labeleihin:

```yaml
income_statement:
  "Toimintatuotot": "operating_income"
  "Toimintakulut": "operating_expenses"
  "Toimintakate": "operating_margin"
  ...
```

---

## Tunnetut rajoitukset

1. **Skannatut PDF:t**: RapidOCR tunnistaa tekstin, mutta EI taulukoita. Taulukot pitää käsitellä erikseen.

2. **RapidOCR GPU**: RapidOCR ei tue CUDA:a suoraan Windowsilla. Käyttää CPU:ta, mutta on silti riittävän nopea.

3. **Suomen kieli**: EasyOCR ei tue suomea, siksi käytetään RapidOCR:ää (kiinankielinen malli, mutta toimii latinalaisille kirjaimille).

---

## Jatkokehitys (TODO)

- [x] ~~Taulukoiden tunnistus skannatuista PDF:istä~~ → PDF_Parser (PP-StructureV3)
- [x] ~~Hybridi-haku (BM25 + vektori)~~ → BGE-M3 + FAISS
- [x] ~~LLM-vastausten generointi~~ → Qwen + Lapua-LLM LoRA
- [ ] Rerank-vaihe (cross-encoder)
- [ ] Reconciliation-testit (summat täsmäävät)
- [ ] Validointi-UI ihmisen tarkistukseen
- [ ] Kysymyspatteriston evaluointi
- [ ] Vertailu vuosien välillä
- [ ] API-rajapinta (FastAPI)

---

## Tekniset muistiinpanot

### Virtuaaliympäristöt

Projektissa on kaksi virtuaaliympäristöä:

| Ympäristö | Polku | Käyttötarkoitus |
|-----------|-------|-----------------|
| `venv` | `./venv/` | CPU-only, perusriippuvuudet |
| `venv_gpu` | `./venv_gpu/` | GPU-tuettu, PyTorch CUDA + ONNX Runtime GPU |

**Aktivointi** (PowerShell):
```powershell
.\venv_gpu\Scripts\Activate.ps1
```

### GPU-komponentit

| Komponentti | Versio | CUDA |
|-------------|--------|------|
| PyTorch | 2.6.0+cu124 | 12.4 |
| Torchvision | 0.21.0+cu124 | 12.4 |
| Sentence-Transformers | latest | PyTorch CUDA |
| FAISS | faiss-cpu | CPU (riittää) |
| Transformers | latest | GPU (4-bit kvantisaatio) |
| PEFT | latest | LoRA-tuki |

### Embedding-malli

Käytetään `BAAI/bge-m3` -mallia:
- Monikielinen (100+ kieltä, suomi ✅)
- 1024-dim vektorit
- 8192 token konteksti-ikkuna
- ~568M parametria
- ~1.5GB muistia GPU:lla

### LLM-malli

Käytetään `Qwen/Qwen2.5-1.5B-Instruct` + LoRA-adapteria:
- Pohjamalli: 1.5B parametria
- LoRA-adapteri: CCG-FAKTUM/lapua-llm-v2 (~10M param)
- 4-bit kvantisaatio (BitsAndBytes)
- ~2GB GPU-muistia

---

## Lisenssi

Sisäinen projekti. Lähdemateriaalit ovat Lapuan kaupungin julkisia asiakirjoja.
