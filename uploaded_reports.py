"""In-memory multi-PDF extraction for Streamlit uploads."""

from __future__ import annotations

from typing import Any

import pymupdf

from extract_learning_journey import (
    chronological_key,
    discover_student_tokens,
    process_document,
)


FORMAT_LIMITATIONS = [
    "Metadata detection recognises common labels such as Foundation, Prep, Year/Grade 1–4, semester and term labels.",
    "Table extraction depends on PDF-native text and line geometry recognised by PyMuPDF.",
    "Graphical ratings are detected only when a row contains 8–12 evenly spaced vector cells matching the developed report layouts.",
    "Current markers are recognised as filled dark circles; previous markers as hollow rings; non-white filled cells may be treated as reference positions.",
    "Curriculum mapping recognises nearby labels formatted like ‘Level 2’. Ambiguous mappings remain unset and flagged.",
    "Scanned image-only pages are not OCR-processed and may yield little or no learning text.",
]


def extract_uploaded_reports(
    uploads: list[tuple[str, bytes]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Process uploaded bytes without writing PDF files to disk."""
    shared_tokens: set[str] = set()
    readable: list[tuple[int, str, bytes]] = []
    summaries: list[dict[str, Any]] = []

    for index, (name, content) in enumerate(uploads, 1):
        try:
            with pymupdf.open(stream=content, filetype="pdf") as document:
                if document.page_count < 1:
                    raise ValueError("empty PDF")
                shared_tokens.update(discover_student_tokens(document))
            readable.append((index, name, content))
        except Exception:
            summaries.append(
                {
                    "upload": index,
                    "status": "not_processed",
                    "pages": 0,
                    "text_sections": 0,
                    "tables": 0,
                    "graphical_ratings": 0,
                    "flags": ["The upload could not be opened as a readable PDF."],
                }
            )

    reports = []
    for index, name, content in readable:
        try:
            with pymupdf.open(stream=content, filetype="pdf") as document:
                report = process_document(document, name, shared_tokens)
            reports.append(report)
            summaries.append(
                {
                    "upload": index,
                    "status": "processed_with_flags" if report["extraction_flags"] else "processed",
                    "pages": report["page_count"],
                    "text_sections": len(report["text_sections"]),
                    "tables": len([table for table in report["tables"] if table.get("rows")]),
                    "graphical_ratings": len(report["graphical_ratings"]),
                    "flags": report["extraction_flags"],
                }
            )
        except Exception as error:
            summaries.append(
                {
                    "upload": index,
                    "status": "partially_processed",
                    "pages": 0,
                    "text_sections": 0,
                    "tables": 0,
                    "graphical_ratings": 0,
                    "flags": [f"Extraction stopped safely ({type(error).__name__}); no values were guessed."],
                }
            )

    if not reports:
        raise ValueError("None of the uploaded files produced a structured report.")
    reports.sort(key=chronological_key)
    for index, report in enumerate(reports, 1):
        report["report_id"] = f"uploaded_{index:02d}"

    dataset = {
        "schema_version": "1.0-uploaded",
        "privacy": {
            "local_only": True,
            "source_pdfs_persisted": False,
            "identifiers_redacted_during_extraction": True,
        },
        "report_count": len(reports),
        "reports": reports,
    }
    summaries.sort(key=lambda item: item["upload"])
    return dataset, summaries
