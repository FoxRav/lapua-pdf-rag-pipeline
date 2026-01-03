"""Generate batch processing report for 25 PDFs."""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

def analyze_parsed_documents():
    """Analyze all parsed documents and generate report."""
    parsed_dir = Path('data/out/parsed')
    results = []
    
    for doc_dir in sorted(parsed_dir.iterdir()):
        if not doc_dir.is_dir():
            continue
            
        doc_id = doc_dir.name
        chunks_file = doc_dir / 'chunks.jsonl'
        doc_file = doc_dir / 'document.jsonl'
        
        chunk_count = 0
        page_count = 0
        text_length = 0
        element_count = 0
        
        if chunks_file.exists():
            with open(chunks_file, 'r', encoding='utf-8') as f:
                for line in f:
                    chunk_count += 1
                    try:
                        chunk = json.loads(line)
                        text_length += len(chunk.get('text', ''))
                    except:
                        pass
        
        if doc_file.exists():
            with open(doc_file, 'r', encoding='utf-8') as f:
                for line in f:
                    element_count += 1
                    try:
                        elem = json.loads(line)
                        if 'page' in elem:
                            page_count = max(page_count, elem['page'])
                    except:
                        pass
        
        results.append({
            'doc_id': doc_id,
            'chunks': chunk_count,
            'pages': page_count,
            'elements': element_count,
            'text_chars': text_length,
            'status': 'OK' if chunk_count > 0 else 'EMPTY'
        })
    
    return results


def analyze_table_chunks():
    """Analyze table chunks from Lapua 2024."""
    tables_file = Path('data/out/2024/tables.jsonl')
    
    if not tables_file.exists():
        return {'count': 0, 'tables': 0}
    
    chunk_count = 0
    table_ids = set()
    
    with open(tables_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunk_count += 1
            try:
                chunk = json.loads(line)
                if 'table_id' in chunk:
                    table_ids.add(chunk['table_id'])
            except:
                pass
    
    return {'count': chunk_count, 'tables': len(table_ids)}


def analyze_complete_index():
    """Analyze complete index."""
    index_dir = Path('data/out/complete_index')
    
    version_file = index_dir / 'version.json'
    meta_file = index_dir / 'chunks_metadata.json'
    
    result = {
        'exists': index_dir.exists(),
        'chunks': 0,
        'version': None
    }
    
    if version_file.exists():
        with open(version_file, 'r', encoding='utf-8') as f:
            result['version'] = json.load(f)
    
    if meta_file.exists():
        with open(meta_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            result['chunks'] = len(meta)
    
    return result


def main():
    print("=" * 70)
    print("25 PDF BATCH RAPORTTI")
    print(f"Aika: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Analyze parsed documents
    docs = analyze_parsed_documents()
    
    print("\n## PROSESSOIDUT DOKUMENTIT")
    print("-" * 70)
    print(f"{'Dokumentti':<40} | {'Chunkit':>7} | {'Sivut':>5} | {'Status':>6}")
    print("-" * 70)
    
    total_chunks = 0
    total_pages = 0
    ok_count = 0
    empty_count = 0
    
    for d in docs:
        short_id = d['doc_id'][:38]
        print(f"{short_id:<40} | {d['chunks']:>7} | {d['pages']:>5} | {d['status']:>6}")
        total_chunks += d['chunks']
        total_pages += d['pages']
        if d['status'] == 'OK':
            ok_count += 1
        else:
            empty_count += 1
    
    print("-" * 70)
    print(f"YHTEENSÄ: {len(docs)} dokumenttia, {total_chunks} chunkkia, {total_pages} sivua")
    print(f"OK: {ok_count}, EMPTY: {empty_count}")
    
    # Table chunks
    tables = analyze_table_chunks()
    print("\n## TAULUKKOCHUNKIT (Lapua 2024)")
    print("-" * 70)
    print(f"Taulukko-chunkkeja: {tables['count']}")
    print(f"Uniikkeja taulukoita: {tables['tables']}")
    
    # Complete index
    index = analyze_complete_index()
    print("\n## COMPLETE INDEX")
    print("-" * 70)
    print(f"Olemassa: {index['exists']}")
    print(f"Chunkkeja indeksissä: {index['chunks']}")
    if index['version']:
        print(f"Versio: {json.dumps(index['version'], indent=2)}")
    
    # Summary
    print("\n## YHTEENVETO")
    print("-" * 70)
    print(f"Tekstichunkit (25 PDF): {total_chunks}")
    print(f"Taulukkochunkit (Lapua 2024): {tables['count']}")
    print(f"Indeksin koko: {index['chunks']}")
    print(f"Odotettu: {total_chunks + tables['count']}")
    
    match = total_chunks + tables['count'] == index['chunks']
    print(f"\nINDEKSI SYNKASSA: {'✅ KYLLÄ' if match else '❌ EI'}")
    
    # Save report as JSON
    report = {
        'timestamp': datetime.now().isoformat(),
        'documents': docs,
        'table_chunks': tables,
        'complete_index': index,
        'summary': {
            'total_docs': len(docs),
            'total_text_chunks': total_chunks,
            'total_table_chunks': tables['count'],
            'total_index_chunks': index['chunks'],
            'ok_docs': ok_count,
            'empty_docs': empty_count,
            'index_synced': match
        }
    }
    
    report_file = Path('eval/batch_report_25pdf.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nRaportti tallennettu: {report_file}")
    
    return report


if __name__ == '__main__':
    main()

