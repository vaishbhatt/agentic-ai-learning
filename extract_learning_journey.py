"""Build a local, de-identified learning-journey dataset from reports/*.pdf.

No OCR, network access, external service, or LLM is used. The source filenames
and identity fields are intentionally omitted from all generated outputs.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import pymupdf
from PIL import Image

from prototype_extract_ratings import candidate_rows, render_row


REPORTS_DIR = Path("reports")
OUTPUT_DIR = Path("learning_journey_output")
VALIDATION_DIR = OUTPUT_DIR / "validation"

GRADE_ORDER = {
    "Kindergarten 1": 0,
    "Kindergarten": 0,
    "Kinder": 0,
    "Foundation": 1,
    "Prep": 1,
    "Year 1": 2,
    "Grade 1": 2,
    "Year 2": 3,
    "Grade 2": 3,
    "Year 3": 4,
    "Grade 3": 4,
    "Year 4": 5,
    "Grade 4": 5,
}


def clean_space(value: str) -> str:
    return " ".join(value.replace("\u00a0", " ").split())


def discover_student_tokens(document: pymupdf.Document) -> set[str]:
    text = "\n".join(page.get_text() for page in document)
    tokens: set[str] = set()
    for match in re.finditer(r"\b([A-Z][a-z]{2,})\s+([A-Z][A-Z'-]{2,})\b", text):
        tokens.update(match.groups())
    possessives = re.findall(r"\b([A-Z][a-z]{2,})['’]s\b", text)
    for token, count in Counter(possessives).items():
        if count >= 1:
            tokens.add(token)
    # Front-page identity lines commonly contain a mixed-case given name and
    # an all-capitals surname. Once the given name is known from possessive
    # references, capture the other alphabetic token on that short line.
    for line in document[0].get_text().splitlines():
        words = re.findall(r"[A-Za-z][A-Za-z'-]+", line)
        if len(words) <= 5 and any(word.lower() in {token.lower() for token in tokens} for word in words):
            tokens.update(word for word in words if len(word) >= 3)
    return tokens


def redact(value: str, student_tokens: set[str]) -> str:
    value = clean_space(value)
    value = re.sub(
        r"\b[A-Z][A-Za-z'&.-]*(?:\s+[A-Z][A-Za-z'&.-]*){0,6}\s+(?:Primary\s+)?School\b",
        "[SCHOOL]",
        value,
    )
    for token in sorted(student_tokens, key=len, reverse=True):
        value = re.sub(rf"\b{re.escape(token)}(?:['’]s)?\b", "[STUDENT]", value, flags=re.I)
    value = re.sub(
        r"\b(?:Mr|Mrs|Ms|Miss|Dr)\.?\s*[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+)?",
        "[STAFF]",
        value,
    )
    value = re.sub(
        r"(?i)\b(?:teacher|observer|principal|coordinator)\s*:\s*[^|,;]+",
        lambda match: match.group(0).split(":", 1)[0] + ": [STAFF]",
        value,
    )
    value = re.sub(r"(?i)\bstudent\s*(?:name|id|number)?\s*:\s*[^|,;]+", "Student: [REDACTED]", value)
    value = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[EMAIL REDACTED]", value)
    value = re.sub(r"\b(?:\+?61|0)[\d ()-]{7,}\b", "[PHONE REDACTED]", value)
    return value.strip()


def is_identity_only(value: str) -> bool:
    lower = value.lower()
    if not value:
        return True
    if "primary school" in lower or re.search(r"\bschool\s*:", lower):
        return True
    if re.search(r"(?i)\b(student|teacher|observer|principal|coordinator)\s*:", value):
        return True
    if re.fullmatch(r"(?:[A-Z][A-Za-z'-]+\s+){1,3}[A-Z][A-Za-z'-]+", value):
        return True
    if re.fullmatch(r"[A-Z]{1,4}\d?[A-Z]?", value):
        return True
    return False


def extract_metadata(document: pymupdf.Document, source_name: str) -> dict[str, Any]:
    page_one = document[0].get_text()
    full_text = "\n".join(page.get_text() for page in document)
    metadata_text = page_one + "\n" + Path(source_name).stem

    years = re.findall(r"\b20\d{2}\b", metadata_text)
    year = int(years[0]) if years else None

    grade_patterns = (
        r"\bKindergarten\s*1\b",
        r"\bKindergarten\b",
        r"\bKinder\b",
        r"\bFoundation\b",
        r"\bPrep\b",
        r"\b(?:Year|Grade)\s*[1-4]\b",
    )
    grade = None
    for search_text in (page_one, full_text):
        if grade:
            break
        for pattern in grade_patterns:
            match = re.search(pattern, search_text, flags=re.I)
            if match:
                raw = clean_space(match.group(0))
                grade = raw.title().replace("Year ", "Year ").replace("Grade ", "Grade ")
                if raw.lower() in {"foundation", "prep", "kinder", "kindergarten"}:
                    grade = raw.capitalize()
                if raw.lower() == "kindergarten 1":
                    grade = "Kindergarten 1"
                break

    period_match = re.search(
        r"\b(?:Semester\s*(?:One|Two|1|2)|Term\s*(?:One|Two|Three|Four|[1-4])|Final Assessment Report)\b",
        page_one,
        flags=re.I,
    )
    if not period_match:
        period_match = re.search(
            r"\b(?:Semester\s*(?:One|Two|1|2)|Term\s*(?:One|Two|Three|Four|[1-4])|Final Assessment Report)\b",
            full_text,
            flags=re.I,
        )
    period = clean_space(period_match.group(0)) if period_match else None

    return {"year": year, "grade": grade, "reporting_period": period}


def classify_text(value: str) -> str:
    lower = value.lower()
    if any(term in lower for term in ("goal", "areas for improvement", "what you can do at home", "next step")):
        return "learning_goal_or_next_step"
    if any(term in lower for term in ("teacher comment", "general comment", "observation")):
        return "teacher_comment_or_observation"
    if any(term in lower for term in ("effort", "behaviour", "behavior", "approaches to learning", "personal learning", "interpersonal")):
        return "learning_disposition_or_personal_indicator"
    if any(term in lower for term in ("achievement", "curriculum", "expected level", "progress")):
        return "assessment_context"
    return "learning_content"


def extract_text_sections(
    page: pymupdf.Page, student_tokens: set[str]
) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        lines = []
        sizes = []
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", []))
            if clean_space(text):
                lines.append(clean_space(text))
                sizes.extend(float(span.get("size", 0)) for span in line.get("spans", []))
        value = redact("\n".join(lines), student_tokens)
        if is_identity_only(value):
            continue
        # Skip page furniture containing no meaningful prose or assessment label.
        bbox = pymupdf.Rect(block["bbox"])
        if bbox.y1 > page.rect.height - 12 and len(value) < 80:
            continue
        sections.append(
            {
                "type": classify_text(value),
                "text": value,
                "page": page.number + 1,
                "bbox": [round(number, 2) for number in block["bbox"]],
                "max_font_size": round(max(sizes, default=0), 2),
                "confidence": 0.98,
            }
        )
    return sections


def extract_tables(page: pymupdf.Page, student_tokens: set[str]) -> list[dict[str, Any]]:
    results = []
    try:
        tables = page.find_tables().tables
    except Exception as error:
        return [{"page": page.number + 1, "status": "failed", "reason": type(error).__name__}]
    for table_index, table in enumerate(tables, 1):
        rows = []
        for raw_row in table.extract():
            row = [redact(cell or "", student_tokens) for cell in raw_row]
            if any(row):
                rows.append(row)
        if rows:
            results.append(
                {
                    "page": page.number + 1,
                    "table": table_index,
                    "bbox": [round(value, 2) for value in table.bbox],
                    "row_count": table.row_count,
                    "column_count": table.col_count,
                    "rows": rows,
                    "confidence": 0.85,
                }
            )
    return results


def dark_boundary_scores(image: Image.Image, boxes: list[tuple[int, int, int, int]]) -> list[int]:
    grey = image.convert("L")
    half_width = max(5, round((boxes[0][2] - boxes[0][0]) * 0.22))
    top = max(0, round(image.height * 0.12))
    bottom = min(image.height, round(image.height * 0.88))
    scores = []
    for box in boxes:
        center_x = box[2]
        crop = grey.crop((max(0, center_x - half_width), top, min(image.width, center_x + half_width + 1), bottom))
        scores.append(sum(crop.histogram()[:80]))
    return scores


def circle_boundary_features(
    image: Image.Image, boxes: list[tuple[int, int, int, int]]
) -> list[tuple[int, int]]:
    grey = image.convert("L")
    center_y = image.height // 2
    features = []
    for box in boxes:
        center_x = box[2]
        center_dark = 0
        ring_dark = 0
        for y in range(max(0, center_y - 14), min(image.height, center_y + 15)):
            for x in range(max(0, center_x - 14), min(image.width, center_x + 15)):
                if grey.getpixel((x, y)) >= 80:
                    continue
                distance = math.hypot(x - center_x, y - center_y)
                if distance <= 4:
                    center_dark += 1
                elif 7 <= distance <= 13:
                    ring_dark += 1
        features.append((center_dark, ring_dark))
    return features


def row_label(page: pymupdf.Page, row: list[Any], student_tokens: set[str]) -> str | None:
    y0, y1 = row[0].rect.y0, row[0].rect.y1
    words = []
    for word in page.get_text("words"):
        x0, wy0, x1, wy1, text = word[:5]
        center_y = (wy0 + wy1) / 2
        if x1 <= row[0].rect.x0 - 2 and y0 - 2 <= center_y <= y1 + 2:
            words.append((wy0, x0, text))
    label = redact(" ".join(item[2] for item in sorted(words)), student_tokens)
    return label or None


def curriculum_labels(page: pymupdf.Page, row: list[Any]) -> list[dict[str, Any]]:
    labels = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = clean_space("".join(span.get("text", "") for span in line.get("spans", [])))
            if not re.fullmatch(r"Level\s+[A-Za-z0-9]+", text, flags=re.I):
                continue
            rect = pymupdf.Rect(line["bbox"])
            if rect.y1 <= row[0].rect.y0 and row[0].rect.y0 - rect.y1 <= 180:
                labels.append({"label": text, "x": round((rect.x0 + rect.x1) / 2, 2)})
    return labels


def nearest_curriculum_label(
    labels: list[dict[str, Any]], marker_x: float, cell_width: float
) -> tuple[str | None, str]:
    if not labels:
        return None, "not_available"
    ranked = sorted((abs(item["x"] - marker_x), item["label"]) for item in labels)
    if len(ranked) > 1 and abs(ranked[0][0] - ranked[1][0]) < cell_width * 0.20:
        return None, "ambiguous_between_labels"
    if ranked[0][0] > cell_width * 1.25:
        return None, "marker_not_close_to_label_anchor"
    return ranked[0][1], "nearest_label_anchor"


def extract_graphical_ratings(
    page: pymupdf.Page, student_tokens: set[str]
) -> list[dict[str, Any]]:
    output = []
    for row_number, row in enumerate(candidate_rows(page), 1):
        image, boxes = render_row(page, row)
        scores = dark_boundary_scores(image, boxes)
        circle_features = circle_boundary_features(image, boxes)
        selected = max(range(len(row)), key=lambda index: circle_features[index][0])
        ranked = sorted(scores, reverse=True)
        margin = ranked[0] - ranked[1]
        excess = ranked[0] - median(scores)
        marker_found = margin > 30 and excess > 50
        previous_candidates = [
            index
            for index, (center_dark, ring_dark) in enumerate(circle_features)
            if index != selected and ring_dark >= 60 and center_dark < 35
        ]
        previous = (
            max(previous_candidates, key=lambda index: circle_features[index][1])
            if previous_candidates
            else None
        )

        fill_distances = [
            math.sqrt(sum((component - 1.0) ** 2 for component in cell.fill)) for cell in row
        ]
        reference = max(range(len(row)), key=fill_distances.__getitem__)
        labels = curriculum_labels(page, row)
        current_x = row[selected].rect.x1
        current_level, level_method = nearest_curriculum_label(
            labels, current_x, row[0].rect.width
        )
        reference_x = row[reference].rect.x1
        reference_level, reference_method = nearest_curriculum_label(
            labels, reference_x, row[0].rect.width
        )

        confidence = round(
            0.4
            + 0.35 * min(1.0, max(0, margin) / 100)
            + 0.25 * min(1.0, max(0, excess) / 150),
            3,
        )
        flags = []
        if not marker_found:
            flags.append("current graphical marker is not distinct enough")
            confidence = min(confidence, 0.49)
        if level_method.startswith("ambiguous") or level_method.startswith("marker_not"):
            flags.append("curriculum level label mapping is ambiguous")

        output.append(
            {
                "page": page.number + 1,
                "row": row_number,
                "assessment_area": row_label(page, row, student_tokens),
                "number_of_positions": len(row),
                "current_position": selected + 1 if marker_found else None,
                "previous_position": previous + 1 if previous is not None else None,
                "progression": (selected - previous) if marker_found and previous is not None else None,
                "reference_position": reference + 1,
                "curriculum_level": current_level,
                "curriculum_level_mapping": level_method,
                "reference_level": reference_level,
                "reference_level_mapping": reference_method,
                "marker_source": "rendered_filled_circle_with_pdf_vector_cell_geometry",
                "confidence": confidence,
                "flags": flags,
            }
        )
    return output


def chronological_key(report: dict[str, Any]) -> tuple[int, int, str]:
    metadata = report["metadata"]
    grade_rank = GRADE_ORDER.get(metadata.get("grade"), 99)
    year = metadata.get("year") or (1900 + grade_rank)
    period = (metadata.get("reporting_period") or "").lower()
    period_rank = 2 if any(value in period for value in ("two", "2", "final")) else 1
    return year, period_rank, metadata.get("grade") or ""


def process_document(
    document: pymupdf.Document,
    source_name: str,
    shared_student_tokens: set[str] | None = None,
) -> dict[str, Any]:
    """Extract one already-open document without storing its source PDF."""
    student_tokens = discover_student_tokens(document) | (shared_student_tokens or set())
    report = {
        "metadata": extract_metadata(document, source_name),
        "page_count": document.page_count,
        "text_sections": [],
        "tables": [],
        "graphical_ratings": [],
        "extraction_flags": [],
    }
    for page in document:
        report["text_sections"].extend(extract_text_sections(page, student_tokens))
        report["tables"].extend(extract_tables(page, student_tokens))
        report["graphical_ratings"].extend(extract_graphical_ratings(page, student_tokens))

    if not report["metadata"]["year"]:
        report["extraction_flags"].append("calendar year was not available confidently")
    if not report["metadata"]["grade"]:
        report["extraction_flags"].append("year/grade was not available confidently")
    if not report["metadata"]["reporting_period"]:
        report["extraction_flags"].append("reporting period was not available confidently")
    if not report["graphical_ratings"]:
        report["extraction_flags"].append(
            "no supported ten-position graphical assessment scale was detected"
        )
    return report


def process_report(source: Path, shared_student_tokens: set[str] | None = None) -> dict[str, Any]:
    with pymupdf.open(source) as document:
        return process_document(document, source.name, shared_student_tokens)


def validation_markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    lines = [
        f"# {report['report_id']} validation",
        "",
        f"- Year: {metadata.get('year') or '[uncertain]'}",
        f"- Grade: {metadata.get('grade') or '[uncertain]'}",
        f"- Reporting period: {metadata.get('reporting_period') or '[uncertain]'}",
        f"- Pages: {report['page_count']}",
        f"- Learning text blocks: {len(report['text_sections'])}",
        f"- Structured tables: {len(report['tables'])}",
        f"- Graphical rating rows: {len(report['graphical_ratings'])}",
        "",
        "## Flags",
        "",
    ]
    flags = report["extraction_flags"] + [
        flag
        for rating in report["graphical_ratings"]
        for flag in rating.get("flags", [])
    ]
    lines.extend(f"- {flag}" for flag in flags or ["No report-level extraction flags."])
    lines.extend(["", "## Graphical ratings", ""])
    if report["graphical_ratings"]:
        for rating in report["graphical_ratings"]:
            lines.append(
                f"- Page {rating['page']}, {rating.get('assessment_area') or 'unlabelled row'}: "
                f"current={rating.get('current_position')}, previous={rating.get('previous_position')}, "
                f"reference={rating.get('reference_position')}, positions={rating['number_of_positions']}, "
                f"level={rating.get('curriculum_level') or '[uncertain]'}, confidence={rating['confidence']}"
            )
    else:
        lines.append("- None detected.")

    lines.extend(["", "## Extracted learning content", ""])
    for section in report["text_sections"]:
        lines.append(f"### Page {section['page']} — {section['type']}")
        lines.extend(["", section["text"], ""])

    lines.extend(["", "## Extracted tables", ""])
    for table in report["tables"]:
        if "rows" not in table:
            lines.append(f"- Page {table['page']}: table extraction {table['status']}.")
            continue
        lines.append(f"### Page {table['page']} table {table['table']}")
        lines.append("")
        for row in table["rows"]:
            lines.append("- " + " | ".join(cell or "[blank]" for cell in row))
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    sources = sorted(REPORTS_DIR.glob("*.pdf"))
    if not sources:
        raise SystemExit("No PDF files found in reports/")
    OUTPUT_DIR.mkdir(exist_ok=True)
    VALIDATION_DIR.mkdir(exist_ok=True)

    shared_student_tokens: set[str] = set()
    for source in sources:
        with pymupdf.open(source) as document:
            shared_student_tokens.update(discover_student_tokens(document))
    reports = [process_report(source, shared_student_tokens) for source in sources]
    reports.sort(key=chronological_key)
    for index, report in enumerate(reports, 1):
        report["report_id"] = f"report_{index:02d}"

    dataset = {
        "schema_version": "1.0-prototype",
        "privacy": {
            "local_only": True,
            "ocr_used": False,
            "external_services_used": False,
            "identifiers_excluded": True,
        },
        "report_count": len(reports),
        "reports": reports,
    }
    dataset_path = OUTPUT_DIR / "learning_journey.json"
    dataset_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")

    index_lines = ["# Learning journey extraction validation", ""]
    for report in reports:
        name = f"{report['report_id']}_validation.md"
        (VALIDATION_DIR / name).write_text(validation_markdown(report), encoding="utf-8")
        metadata = report["metadata"]
        index_lines.append(
            f"- [{report['report_id']}](validation/{name}): "
            f"{metadata.get('year') or 'year uncertain'}, "
            f"{metadata.get('grade') or 'grade uncertain'}, "
            f"{metadata.get('reporting_period') or 'period uncertain'}"
        )
    validation_index = OUTPUT_DIR / "validation_summary.md"
    validation_index.write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    print(f"reports_processed={len(reports)}")
    print(f"dataset={dataset_path}")
    print(f"validation_index={validation_index}")
    print(f"validation_reports={VALIDATION_DIR}")
    print(f"reports_with_flags={sum(bool(report['extraction_flags']) for report in reports)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
