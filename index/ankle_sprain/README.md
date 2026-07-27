# Ankle Sprain RAG Index

This directory contains the extracted English text from 32 trusted
ankle-sprain PDFs: patient resources from hospitals, government health
services, and professional medical organizations, plus peer-reviewed research
indexed in PubMed Central. The source content has not been translated or
paraphrased.

## Files

- `chunks.jsonl`: one JSON object per chunk, suitable for embedding or direct
  loading into a RAG pipeline.
- `chunks.sqlite3`: the same chunks plus a local SQLite FTS5 full-text index.
- `embeddings/embeddings.npz`: normalized dense vectors in the same order as
  the stored `chunk_ids` array.
- `embeddings/metadata.json`: model, dimensions, checksums, and generation
  settings for the vectors.

Each chunk contains its source filename, issuing institution, official URL,
document and review dates, page range, character count, and original text.

## Rebuild

Run the ingestion script from the project root:

```bash
python ingest.py
```

Install `pypdf` first if it is not already available:

```bash
python -m pip install pypdf
```

## Example full-text query

```sql
SELECT
    c.chunk_id,
    c.filename,
    c.page_start,
    c.page_end,
    c.text
FROM chunks_fts AS f
JOIN chunks AS c USING (chunk_id)
WHERE chunks_fts MATCH 'balance exercise'
ORDER BY bm25(chunks_fts)
LIMIT 5;
```

The SQLite index is lexical rather than semantic. For vector retrieval, generate
embeddings from the `text` field in `chunks.jsonl` while retaining all metadata.

Run `python embed.py` from the project root to build the local semantic index.
