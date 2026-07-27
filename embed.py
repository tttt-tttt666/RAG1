#!/usr/bin/env python3
"""Generate normalized local embeddings for the extracted RAG chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent
CHUNKS_PATH = ROOT / "index" / "ankle_sprain" / "chunks.jsonl"
OUTPUT_DIR = ROOT / "index" / "ankle_sprain" / "embeddings"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_chunks() -> list[dict]:
    with CHUNKS_PATH.open(encoding="utf-8") as stream:
        chunks = [json.loads(line) for line in stream if line.strip()]
    if not chunks:
        raise ValueError(f"No chunks found in {CHUNKS_PATH}")
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    chunks = load_chunks()
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    texts = [chunk["text"] for chunk in chunks]

    model = SentenceTransformer(args.model)
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    if embeddings.shape[0] != len(chunks):
        raise ValueError("Embedding count does not match chunk count")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain non-finite values")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vectors_path = OUTPUT_DIR / "embeddings.npz"
    metadata_path = OUTPUT_DIR / "metadata.json"

    np.savez_compressed(
        vectors_path,
        embeddings=embeddings,
        chunk_ids=np.asarray(chunk_ids),
    )

    norms = np.linalg.norm(embeddings, axis=1)
    metadata = {
        "model": args.model,
        "embedding_dimension": int(embeddings.shape[1]),
        "embedding_count": int(embeddings.shape[0]),
        "dtype": str(embeddings.dtype),
        "normalized": True,
        "minimum_l2_norm": float(norms.min()),
        "maximum_l2_norm": float(norms.max()),
        "source_chunks": str(CHUNKS_PATH.relative_to(ROOT)),
        "source_chunks_sha256": sha256(CHUNKS_PATH),
        "vectors_file": vectors_path.name,
        "vectors_sha256": sha256(vectors_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"Saved {embeddings.shape[0]} normalized embeddings "
        f"with {embeddings.shape[1]} dimensions"
    )
    print(f"Vectors: {vectors_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
