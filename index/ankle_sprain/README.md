# Ankle Sprain RAG Index

This directory contains the extracted English text from the four official
ankle-sprain patient education PDFs. The source content has not been translated
or paraphrased.

## Files

- `chunks.jsonl`: one JSON object per chunk, suitable for embedding or direct
  loading into a RAG pipeline.
- `chunks.sqlite3`: the same chunks plus a local SQLite FTS5 full-text index.

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
