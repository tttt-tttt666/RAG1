#!/usr/bin/env python3
"""Extract, chunk, and index the ankle-sprain education and evidence PDFs.

The source text remains in English. Processing is limited to whitespace
normalization and sentence-aware chunking; no translation or rewriting occurs.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
PDF_DIR = ROOT / "output" / "pdf" / "ankle_sprain_patient_education"
INDEX_DIR = ROOT / "index" / "ankle_sprain"
JSONL_PATH = INDEX_DIR / "chunks.jsonl"
SQLITE_PATH = INDEX_DIR / "chunks.sqlite3"

MIN_CHARS = 500
TARGET_CHARS = 650
MAX_CHARS = 800

SOURCES = {
    "01_PAH_NHS_Ankle_Sprain_Patient_Leaflet.pdf": {
        "institution": "The Princess Alexandra Hospital NHS Trust",
        "source_url": "https://www.pah.nhs.uk/wp-content/uploads/2025/09/Ankle_Sprain_PIL_v2.pdf",
        "document_date": "2025-08",
        "review_date": "2028-08",
    },
    "02_UHSussex_NHS_Ankle_Sprain_AE_Leaflet.pdf": {
        "institution": "University Hospitals Sussex NHS Foundation Trust",
        "source_url": "https://www.uhsussex.nhs.uk/wp-content/uploads/2018/05/888.2-Ankle-Sprain-AE-leaflet-2025.pdf",
        "document_date": "2025-07",
        "review_date": "2028-07",
    },
    "03_Berkshire_NHS_Ankle_Sprains_Rehab.pdf": {
        "institution": "Berkshire Healthcare NHS Foundation Trust",
        "source_url": "https://www.berkshirehealthcare.nhs.uk/media/vvvhk1eq/bh1165b-ankle-sprains-v1-feb-2026.pdf",
        "document_date": "2026-02",
        "review_date": "2028-02",
    },
    "04_Sherwood_Forest_NHS_Ankle_Sprains_and_Strains.pdf": {
        "institution": "Sherwood Forest Hospitals NHS Foundation Trust",
        "source_url": "https://sfh-tr.nhs.uk/media/k5ghz1jb/ankle-sprains-and-strains.pdf",
        "document_date": "2026-03",
        "review_date": "2028-03",
    },
    "05_PMC8824326_Exercise_Rehabilitation_Systematic_Review.pdf": {
        "institution": "PLOS ONE (indexed in PubMed Central)",
        "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8824326/",
        "document_date": "2022-02-08",
        "review_date": "not applicable",
    },
    "06_PMC9301067_Acute_Ankle_Sprain_Umbrella_Review.pdf": {
        "institution": "Frontiers in Medicine (indexed in PubMed Central)",
        "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9301067/",
        "document_date": "2022-07-07",
        "review_date": "not applicable",
    },
    "07_PMC12481793_Physical_Therapy_Meta_Analysis.pdf": {
        "institution": "BMC Musculoskeletal Disorders (indexed in PubMed Central)",
        "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12481793/",
        "document_date": "2025-09-29",
        "review_date": "not applicable",
    },
    "08_Alder_Hey_NHS_Ankle_Foot_Sprain.pdf": {
        "institution": "Alder Hey Children's NHS Foundation Trust",
        "source_url": "https://www.alderhey.nhs.uk/wp-content/uploads/2024/03/PIAG-523-Ankle-Foot-Sprain.pdf",
        "document_date": "not stated",
        "review_date": "2027-02",
    },
    "09_Plymouth_NHS_Sprained_Ankle_Foot.pdf": {
        "institution": "University Hospitals Plymouth NHS Trust",
        "source_url": "https://www.plymouthhospitals.nhs.uk/download/sprained-ankle-foot-final-february-2025-v2pdf.pdf?doc=docm93jijm4n19511.pdf&ver=30493&ver=30533",
        "document_date": "2025-02",
        "review_date": "2027-02",
    },
    "10_ULH_NHS_Ankle_Sprain_Exercises.pdf": {
        "institution": "United Lincolnshire Hospitals NHS Trust",
        "source_url": "https://www.ulh.nhs.uk/wp-content/uploads/2025/07/Ankle-Sprain-Exercises.pdf",
        "document_date": "2025-06",
        "review_date": "2027-06",
    },
    "11_Oxford_NHS_Ankle_Sprain_Advice.pdf": {
        "institution": "Oxford University Hospitals NHS Foundation Trust",
        "source_url": "https://www.ouh.nhs.uk/media/i0ldiecy/116454sprain.pdf",
        "document_date": "2025-10",
        "review_date": "2028-10",
    },
    "12_Mersey_West_Lancs_NHS_Ankle_Sprain.pdf": {
        "institution": "Mersey and West Lancashire Teaching Hospitals NHS Trust",
        "source_url": "https://sthk.merseywestlancs.nhs.uk/media/.leaflets/6593fe207af493.62309476.pdf",
        "document_date": "not stated",
        "review_date": "2026-12-01",
    },
}


@dataclass
class Unit:
    text: str
    page: int


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    filename: str
    institution: str
    source_url: str
    document_date: str
    review_date: str
    page_start: int
    page_end: int
    char_count: int
    text: str


def normalize_text(text: str) -> str:
    """Normalize layout whitespace without translating or paraphrasing."""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n+ *", "\n", text)
    return text.strip()


def split_long_unit(text: str, page: int) -> list[Unit]:
    """Split oversized text at sentence boundaries, then word boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text)
    output: list[Unit] = []
    for sentence in sentences:
        sentence = sentence.strip()
        while len(sentence) > MAX_CHARS:
            cut = sentence.rfind(" ", 0, MAX_CHARS + 1)
            if cut < MIN_CHARS:
                cut = MAX_CHARS
            output.append(Unit(sentence[:cut].strip(), page))
            sentence = sentence[cut:].strip()
        if sentence:
            output.append(Unit(sentence, page))
    return output


def extract_units(pdf_path: Path) -> list[Unit]:
    reader = PdfReader(str(pdf_path))
    units: list[Unit] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")
        for paragraph in re.split(r"\n+", text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            units.extend(split_long_unit(paragraph, page_number))
    return units


def pack_units(units: list[Unit]) -> list[list[Unit]]:
    """Pack consecutive source units into approximately 500-800 characters."""
    groups: list[list[Unit]] = []
    current: list[Unit] = []
    current_len = 0

    for unit in units:
        added_len = len(unit.text) + (1 if current else 0)
        if current and current_len >= MIN_CHARS and current_len + added_len > TARGET_CHARS:
            groups.append(current)
            current = []
            current_len = 0
        if current and current_len + added_len > MAX_CHARS:
            groups.append(current)
            current = []
            current_len = 0
        current.append(unit)
        current_len += len(unit.text) + (1 if len(current) > 1 else 0)

    if current:
        groups.append(current)

    # Avoid a tiny final chunk when it can be merged without exceeding MAX_CHARS.
    if len(groups) >= 2:
        last_len = len(" ".join(unit.text for unit in groups[-1]))
        previous_len = len(" ".join(unit.text for unit in groups[-2]))
        if last_len < MIN_CHARS and previous_len + 1 + last_len <= MAX_CHARS:
            groups[-2].extend(groups[-1])
            groups.pop()

    return groups


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_chunks() -> tuple[list[Chunk], dict[str, dict[str, str | int]]]:
    chunks: list[Chunk] = []
    documents: dict[str, dict[str, str | int]] = {}

    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        if pdf_path.name not in SOURCES:
            continue
        metadata = SOURCES[pdf_path.name]
        document_id = pdf_path.stem
        reader = PdfReader(str(pdf_path))
        groups = pack_units(extract_units(pdf_path))
        documents[document_id] = {
            "filename": pdf_path.name,
            "institution": metadata["institution"],
            "source_url": metadata["source_url"],
            "document_date": metadata["document_date"],
            "review_date": metadata["review_date"],
            "page_count": len(reader.pages),
            "chunk_count": len(groups),
            "sha256": sha256(pdf_path),
        }
        for number, group in enumerate(groups, start=1):
            text = " ".join(unit.text for unit in group)
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}__chunk_{number:03d}",
                    document_id=document_id,
                    filename=pdf_path.name,
                    institution=metadata["institution"],
                    source_url=metadata["source_url"],
                    document_date=metadata["document_date"],
                    review_date=metadata["review_date"],
                    page_start=min(unit.page for unit in group),
                    page_end=max(unit.page for unit in group),
                    char_count=len(text),
                    text=text,
                )
            )
    return chunks, documents


def write_jsonl(chunks: list[Chunk]) -> None:
    with JSONL_PATH.open("w", encoding="utf-8") as stream:
        for chunk in chunks:
            stream.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def write_sqlite(
    chunks: list[Chunk], documents: dict[str, dict[str, str | int]]
) -> None:
    with sqlite3.connect(SQLITE_PATH) as connection:
        connection.executescript(
            """
            DROP TABLE IF EXISTS chunks_fts;
            DROP TABLE IF EXISTS chunks;
            DROP TABLE IF EXISTS documents;

            CREATE TABLE documents (
                document_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                institution TEXT NOT NULL,
                source_url TEXT NOT NULL,
                document_date TEXT NOT NULL,
                review_date TEXT NOT NULL,
                page_count INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL,
                sha256 TEXT NOT NULL
            );

            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                institution TEXT NOT NULL,
                source_url TEXT NOT NULL,
                document_date TEXT NOT NULL,
                review_date TEXT NOT NULL,
                page_start INTEGER NOT NULL,
                page_end INTEGER NOT NULL,
                char_count INTEGER NOT NULL,
                text TEXT NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(document_id)
            );

            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                chunk_id UNINDEXED,
                text,
                tokenize='porter unicode61'
            );
            """
        )
        connection.executemany(
            """
            INSERT INTO documents VALUES (
                :document_id, :filename, :institution, :source_url,
                :document_date, :review_date, :page_count, :chunk_count, :sha256
            )
            """,
            [
                {"document_id": document_id, **metadata}
                for document_id, metadata in documents.items()
            ],
        )
        rows = [asdict(chunk) for chunk in chunks]
        connection.executemany(
            """
            INSERT INTO chunks VALUES (
                :chunk_id, :document_id, :filename, :institution, :source_url,
                :document_date, :review_date, :page_start, :page_end,
                :char_count, :text
            )
            """,
            rows,
        )
        connection.executemany(
            "INSERT INTO chunks_fts(chunk_id, text) VALUES (:chunk_id, :text)",
            rows,
        )


def main() -> None:
    expected = set(SOURCES)
    actual = {path.name for path in PDF_DIR.glob("*.pdf")}
    missing = expected - actual
    if missing:
        raise FileNotFoundError(f"Missing source PDFs: {sorted(missing)}")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    chunks, documents = build_chunks()
    write_jsonl(chunks)
    write_sqlite(chunks, documents)

    lengths = [chunk.char_count for chunk in chunks]
    print(f"Indexed {len(documents)} documents into {len(chunks)} chunks")
    print(
        f"Chunk characters: min={min(lengths)}, "
        f"median={sorted(lengths)[len(lengths) // 2]}, max={max(lengths)}"
    )
    print(f"JSONL: {JSONL_PATH}")
    print(f"SQLite FTS5: {SQLITE_PATH}")


if __name__ == "__main__":
    main()
