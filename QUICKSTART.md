# 🚀 QUICKSTART - Ohjelman käynnistys

## Kun avaat projektin uudelleen (esim. seuraavana päivänä)

### 1. Avaa PowerShell/Terminal projektin juuressa

```powershell
cd "F:\-DEV-\33.Lapua-tilinpäätös2025-20250201"
```

### 2. Aktivoi GPU-ympäristö

```powershell
.\venv_gpu\Scripts\Activate.ps1
```

Näet `(venv_gpu)` promptin alussa kun aktivointi onnistui.

### 3. Aseta PYTHONPATH ja encoding

```powershell
$env:PYTHONPATH = "."
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

### 4. Valmis! Testaa toimivuus:

```powershell
# Nopea testi - hae jotain indeksistä
python -m src.pipeline.query_complete "Mikä on vuosikate?"
```

---

## Kaikki komennot yhdellä rivillä (kopioi-liitä)

```powershell
cd "F:\-DEV-\33.Lapua-tilinpäätös2025-20250201"; .\venv_gpu\Scripts\Activate.ps1; $env:PYTHONPATH = "."; [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

---

## Tärkeimmät komennot

### Haku (hybridi + reranking)
```powershell
python -m src.pipeline.query_complete "Mikä on toimintakate?"
```

### LLM-vastaus evidenssillä
```powershell
python -m src.pipeline.answer_with_evidence "Paljonko oli poistoja?"
```

### 90 kysymyksen evaluointi
```powershell
# Kaikki 90 kysymystä
python -m eval.run_questions_batch

# Vain pakolliset 20 kysymystä (nopeampi)
python -m eval.run_questions_batch --must-only
```

### Smoke-testit (50 testiä)
```powershell
python -m eval.run_smoke_eval_v2
```

---

## Jos jotain on rikki

### 1. Tarkista GPU
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

Pitäisi tulostaa: `CUDA: True`

### 2. Tarkista indeksi
```powershell
python -c "import json; d=json.load(open('data/out/complete_index/chunks_metadata.json')); print(f'Chunks: {len(d)}')"
```

Pitäisi tulostaa: `Chunks: 1773`

### 3. Tarkista mallit (latautuvat ensimmäisellä kerralla)
- `BAAI/bge-m3` (embedding) - ~1.5GB VRAM
- `BAAI/bge-reranker-v2-m3` (reranker) - ~1GB VRAM  
- `Qwen/Qwen2.5-1.5B-Instruct` + LoRA (LLM) - ~2GB VRAM

---

## Projektitiedot

| Tieto | Arvo |
|-------|------|
| Dokumentteja | 25 |
| Chunkkeja | 1773 (545 text + 1228 table) |
| Indeksi | `data/out/complete_index/` |
| Kysymyksiä | 90 (eval/questions_full_90.json) |
| Smoke-testit | 50 (eval/smoke_2024_full.json) |

---

## Cursor AI:lle muistutus

Kun käyttäjä avaa projektin uudelleen:

1. **Aktivoi ympäristö** ennen mitään komentoja
2. **Aseta PYTHONPATH** aina `.`
3. **Aseta UTF-8 encoding** suomen kielen takia
4. **Indeksi on valmiina** - ei tarvitse rakentaa uudelleen
5. **Mallit latautuvat** ensimmäisellä kyselyllä (~30-60s)

Ympäristön tila säilyy samassa terminaalissa, mutta **uusi terminaali vaatii aina aktivoinnin**.

