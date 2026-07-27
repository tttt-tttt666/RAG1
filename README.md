# Ankle Sprain Patient Education RAG Dataset

This project prepares four official NHS ankle-sprain patient education PDFs for
retrieval-augmented generation (RAG).

The original English medical text is extracted without translation or
paraphrasing, split into approximately 500-800 character chunks, and indexed
with SQLite FTS5.

## Contents

- `output/pdf/ankle_sprain_patient_education/`: original source PDFs and source metadata
- `ingest.py`: reproducible extraction, chunking, and indexing script
- `embed.py`: local semantic embedding generation script
- `index/ankle_sprain/chunks.jsonl`: 26 chunks with source and page metadata
- `index/ankle_sprain/chunks.sqlite3`: local full-text search index
- `index/ankle_sprain/embeddings/`: normalized dense vectors and metadata

## Setup

```bash
python -m pip install -r requirements.txt
python ingest.py
python embed.py
```

See `index/ankle_sprain/README.md` for an example full-text query.

## Medical disclaimer

These materials are for patient education and RAG experimentation only. This
project does not provide diagnosis, individualized rehabilitation plans, or a
substitute for advice from a qualified healthcare professional.
