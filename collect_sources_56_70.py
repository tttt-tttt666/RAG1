#!/usr/bin/env python3
"""Collect official ankle education sources 56-70 as English PDF snapshots.

HTML snapshots preserve the issuing page's English text without translation or
paraphrasing. Original PDFs are downloaded unchanged.
"""

from __future__ import annotations

import html
import re
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

from lxml import etree
from lxml import html as lxml_html
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output" / "pdf" / "ankle_sprain_patient_education"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 Chrome/124 Safari/537.36"
)

SOURCES = [
    {
        "filename": "56_Cleveland_Clinic_Ankle_Anatomy.pdf",
        "title": "Ankle: Anatomy & How It Works",
        "institution": "Cleveland Clinic",
        "date": "2023-04-15",
        "url": "https://my.clevelandclinic.org/health/body/24909-ankle-joint",
        "start": "What is the ankle joint?",
        "end": "A note from Cleveland Clinic",
    },
    {
        "filename": "57_RNOH_Common_Peroneal_Nerve_Guide.pdf",
        "title": "A Patient's Guide to Common Peroneal Nerve Exploration",
        "institution": "Royal National Orthopaedic Hospital NHS Trust",
        "date": "2025",
        "url": "https://www.rnoh.nhs.uk/patients-and-visitors/patient-information-guides/common-peroneal-nerve-exploration-patients-guide",
        "start": "What is the Common Peroneal Nerve?",
        "end": "Contact us",
    },
    {
        "filename": "58_Cleveland_Clinic_Peroneal_Nerve_Injury.pdf",
        "title": "Peroneal Nerve Injury: Symptoms, Causes & Treatment",
        "institution": "Cleveland Clinic",
        "date": "2026-03-31",
        "url": "https://my.clevelandclinic.org/health/diseases/24263-peroneal-nerve-injury",
        "start": "What Is a Peroneal Nerve Injury?",
        "end": "A note from Cleveland Clinic",
    },
    {
        "filename": "59_MedlinePlus_Ankle_Injuries_and_Disorders.pdf",
        "title": "Ankle Injuries and Disorders",
        "institution": "MedlinePlus, U.S. National Library of Medicine",
        "date": "2024-09-30",
        "url": "https://medlineplus.gov/ankleinjuriesanddisorders.html",
        "start": "Summary",
        "end": "Disclaimers",
    },
    {
        "filename": "60_MedlinePlus_Ankle_Pain.pdf",
        "title": "Ankle Pain",
        "institution": "MedlinePlus, U.S. National Library of Medicine",
        "date": "not stated",
        "url": "https://medlineplus.gov/ency/article/003167.htm",
        "start": "Ankle pain",
        "end": "References",
    },
    {
        "filename": "61_Plymouth_NHS_Ankle_Fractures.pdf",
        "title": "Ankle Fractures",
        "institution": "University Hospitals Plymouth NHS Trust",
        "date": "2025-07",
        "url": "https://www.plymouthhospitals.nhs.uk/display-pil/pil-ankle-fractures-3956",
        "start": "Basic Anatomy",
        "end": "Further Information",
    },
    {
        "filename": "62_Plymouth_NHS_Achilles_Rupture_Rehabilitation.pdf",
        "title": "Achilles Tendon Rupture Rehabilitation",
        "institution": "University Hospitals Plymouth NHS Trust",
        "date": "2025-08",
        "url": "https://www.plymouthhospitals.nhs.uk/display-pil/pil-achilles-tendon-rupture-rehabilitation-8516",
        "start": "Important:",
        "end": "Further Information",
    },
    {
        "filename": "63_East_Sussex_NHS_Ankle_Avulsion_Fracture.pdf",
        "title": "Ankle Avulsion Fracture",
        "institution": "East Sussex Healthcare NHS Trust",
        "date": "not stated",
        "url": "https://www.esht.nhs.uk/leaflet/ankle-avulsion-fracture/",
        "start": "Ankle Avulsion Fracture",
        "end": "Contact",
    },
    {
        "filename": "64_MedlinePlus_Ankle_Fracture_Aftercare.pdf",
        "title": "Ankle Fracture - Aftercare",
        "institution": "MedlinePlus, U.S. National Library of Medicine",
        "date": "2024-06-17",
        "url": "https://medlineplus.gov/ency/patientinstructions/000548.htm",
        "start": "Ankle fracture - aftercare",
        "end": "References",
    },
    {
        "filename": "65_North_Tees_NHS_Ankle_Fracture.pdf",
        "title": "Ankle Fracture",
        "institution": "North Tees and Hartlepool NHS Foundation Trust",
        "date": "2026",
        "url": "https://www.nth.nhs.uk/resources/ankle-fracture/",
        "start": "Ankle Fracture",
        "end": "Useful sources of information",
    },
    {
        "filename": "66_Worcestershire_NHS_Ankle_Fracture_Rehabilitation.pdf",
        "title": "Ankle Fracture",
        "institution": "Worcestershire Acute Hospitals NHS Trust",
        "date": "2026-06-23",
        "url": "https://www.worcsacute.nhs.uk/leaflets/ankle-fracture/",
        "start": "Ankle Fracture",
        "end": "Feedback for Inpatient Therapies",
        "raw_start": '<p class="wp-block-paragraph">This leaflet will provide',
        "raw_end": '<p class="wp-block-paragraph"><strong>Feedback&nbsp;for Inpatient Therapies',
    },
    {
        "filename": "67_Imperial_NHS_Achilles_Tendon_Partial_Tear.pdf",
        "title": "Achilles Tendon - Partial Tear",
        "institution": "Imperial College Healthcare NHS Trust",
        "date": "2026",
        "url": "https://www.imperial.nhs.uk/-/media/website/patient-information-leaflets/orthopaedics/virtual-fracture-clinic/achilles-tendon--partial-tear.pdf?rev=ad40df832c2e4a32bbaf964d7ef96f93",
        "kind": "pdf",
    },
    {
        "filename": "68_Doncaster_NHS_Ankle_Fracture_Exercises.pdf",
        "title": "Advice and Exercises for Ankle Fractures",
        "institution": "Doncaster and Bassetlaw Teaching Hospitals NHS Foundation Trust",
        "date": "not stated",
        "url": "https://www.dbth.nhs.uk/advice-and-exercises-for-ankle-fractures/",
        "start": "You have fractured your ankle",
        "end": "Contact",
    },
    {
        "filename": "69_Royal_Cornwall_NHS_Achilles_Tendon_Injury.pdf",
        "title": "Achilles Tendon Injury",
        "institution": "Royal Cornwall Hospitals NHS Trust",
        "date": "2024-10-02",
        "url": "https://fractureclinic.royalcornwallhospitals.nhs.uk/ankle-injuries/achilles-tendon-injury/",
        "start": "The information here will help",
        "end": "Page last reviewed",
    },
    {
        "filename": "70_AOFAS_FootCareMD_Sprained_Ankle_Care.pdf",
        "title": "Sprained Ankle Self Care and Treatment",
        "institution": "American Orthopaedic Foot & Ankle Society (FootCareMD)",
        "date": "not stated",
        "url": "https://www.footcaremd.org/resources/how-to-help/how-to-care-for-a-sprained-ankle",
        "start": "What should I do if I sprain my ankle?",
        "end": "The American Orthopaedic Foot & Ankle Society",
    },
]


def fetch(url: str) -> tuple[bytes, str]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read(), response.headers.get("Content-Type", "")
    except Exception:
        completed = subprocess.run(
            [
                "curl",
                "-L",
                "--fail",
                "--silent",
                "--show-error",
                "-A",
                USER_AGENT,
                url,
            ],
            check=True,
            capture_output=True,
        )
        payload = completed.stdout
        content_type = (
            "application/pdf" if payload.startswith(b"%PDF") else "text/html"
        )
        return payload, content_type


def normalized(value: str) -> str:
    value = value.replace("\xa0", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def extract_blocks(payload: bytes, source: dict[str, str]) -> list[str]:
    document = lxml_html.fromstring(payload)
    candidates = document.xpath("//main|//article")
    root = max(
        [*candidates, document],
        key=lambda node: len(normalized(node.text_content())),
    )
    blocks: list[str] = []
    for element in root.xpath(".//h1|.//h2|.//h3|.//h4|.//p|.//li|.//th|.//td"):
        text = normalized(element.text_content())
        if not text or len(text) < 2:
            continue
        if blocks and blocks[-1] == text:
            continue
        blocks.append(text)

    start = source.get("start", "")
    end = source.get("end", "")
    if start:
        for index, text in enumerate(blocks):
            if start.lower() in text.lower():
                blocks = blocks[index:]
                break
    if end:
        for index, text in enumerate(blocks):
            if index > 0 and end.lower() in text.lower():
                blocks = blocks[:index]
                break

    boilerplate = (
        "cookie",
        "skip to",
        "share this",
        "print this page",
        "advertisement",
        "accept all",
        "privacy preference",
    )
    blocks = [
        text
        for text in blocks
        if not any(item in text.lower() for item in boilerplate)
    ]
    if len(" ".join(blocks)) < 700 and start:
        raw_html = payload.decode("utf-8", errors="ignore")
        raw_lower = raw_html.lower()
        raw_start_marker = source.get("raw_start", start)
        raw_end_marker = source.get("raw_end", end)
        raw_start = raw_lower.find(raw_start_marker.lower())
        if raw_start >= 0:
            raw_end = (
                raw_lower.find(
                    raw_end_marker.lower(),
                    raw_start + len(raw_start_marker),
                )
                if raw_end_marker
                else -1
            )
            fragment_text = raw_html[raw_start : raw_end if raw_end >= 0 else None]
            fragment = lxml_html.fragment_fromstring(
                f"<div>{fragment_text}</div>",
                create_parent=False,
            )
            blocks = []
            for element in fragment.xpath(
                ".//h1|.//h2|.//h3|.//h4|.//p|.//li|.//th|.//td"
            ):
                text = normalized(element.text_content())
                if text and len(text) >= 2 and (not blocks or blocks[-1] != text):
                    blocks.append(text)
    if len(" ".join(blocks)) < 700:
        raise ValueError(
            f"Extracted source text is unexpectedly short ({len(' '.join(blocks))} chars)"
        )
    return blocks


def make_snapshot(source: dict[str, str], blocks: list[str], destination: Path) -> None:
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "SnapshotTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=26,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#17345F"),
        spaceAfter=14,
    )
    meta_style = ParagraphStyle(
        "SnapshotMeta",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#334155"),
        spaceAfter=7,
    )
    body_style = ParagraphStyle(
        "SnapshotBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        spaceAfter=9,
        textColor=colors.HexColor("#111827"),
    )
    heading_style = ParagraphStyle(
        "SnapshotHeading",
        parent=body_style,
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        spaceBefore=8,
        spaceAfter=7,
        textColor=colors.HexColor("#17345F"),
    )

    story = [
        Spacer(1, 20 * mm),
        Paragraph(html.escape(source["title"]), title_style),
        Spacer(1, 5 * mm),
        Paragraph(
            f"<b>Issuing institution:</b> {html.escape(source['institution'])}",
            meta_style,
        ),
        Paragraph(
            f"<b>Publication / update date:</b> {html.escape(source['date'])}",
            meta_style,
        ),
        Paragraph(
            f"<b>Original official URL:</b> {html.escape(source['url'])}",
            meta_style,
        ),
        Spacer(1, 8 * mm),
        Paragraph(
            "Provenance note: This PDF is a local English-text snapshot of the "
            "official page. Navigation was removed. The medical text below was "
            "not translated, paraphrased, or supplemented.",
            meta_style,
        ),
        PageBreak(),
    ]

    for text in blocks:
        escaped = html.escape(text)
        is_heading = len(text) <= 110 and not text.endswith((".", "?", "!"))
        story.append(Paragraph(escaped, heading_style if is_heading else body_style))

    document = SimpleDocTemplate(
        str(destination),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=17 * mm,
        title=source["title"],
        author=source["institution"],
    )
    document.build(story)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        destination = OUTPUT_DIR / source["filename"]
        if destination.exists():
            reader = PdfReader(str(destination))
            extracted = " ".join((page.extract_text() or "") for page in reader.pages)
            if len(extracted) >= 700:
                print(
                    f"{destination.name}: {len(reader.pages)} pages, "
                    f"{len(extracted)} extracted chars (existing)"
                )
                continue
        payload, content_type = fetch(source["url"])
        if source.get("kind") == "pdf":
            if not payload.startswith(b"%PDF"):
                raise ValueError(
                    f"{source['filename']} did not return a PDF ({content_type})"
                )
            destination.write_bytes(payload)
        else:
            blocks = extract_blocks(payload, source)
            make_snapshot(source, blocks, destination)

        reader = PdfReader(str(destination))
        extracted = " ".join((page.extract_text() or "") for page in reader.pages)
        if len(extracted) < 700:
            raise ValueError(
                f"{destination.name} has insufficient extractable text: {len(extracted)}"
            )
        print(
            f"{destination.name}: {len(reader.pages)} pages, "
            f"{len(extracted)} extracted chars"
        )


if __name__ == "__main__":
    main()
