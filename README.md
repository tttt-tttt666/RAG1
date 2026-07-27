# Ankle Sprain Patient Education RAG Dataset

This project prepares 32 trusted ankle-sprain PDFs for retrieval-augmented
generation (RAG). The collection combines official patient education from
hospitals, government health services, and professional medical organizations
with three peer-reviewed evidence syntheses indexed in PubMed Central.

The original English medical text is extracted without translation or
paraphrasing, split into approximately 500-800 character chunks, and indexed
with SQLite FTS5.

## Contents

- `output/pdf/ankle_sprain_patient_education/`: source PDFs, official-page PDF snapshots, and source metadata
- `ingest.py`: reproducible extraction, chunking, and indexing script
- `embed.py`: local semantic embedding generation script
- `index/ankle_sprain/chunks.jsonl`: source chunks with page metadata
- `index/ankle_sprain/chunks.sqlite3`: local full-text search index
- `index/ankle_sprain/embeddings/`: normalized dense vectors and metadata
- `app.py`: local Streamlit interface with short Chinese template answers,
  source passages, and safety warnings

## Setup

```bash
python -m pip install -r requirements.txt
python ingest.py
python embed.py
streamlit run app.py
```

Open `http://localhost:8501` after starting Streamlit.

See `index/ankle_sprain/README.md` for an example full-text query.

## Medical disclaimer

These materials are for patient education and RAG experimentation only. This
project does not provide diagnosis, individualized rehabilitation plans, or a
substitute for advice from a qualified healthcare professional.
