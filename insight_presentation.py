"""Deterministically structure an existing cached insight response for display."""

from __future__ import annotations

import re
from typing import Any


SECTION_PATTERN = re.compile(r"^(\d+)\)\s+(.+)$")
PERIOD_PATTERN = re.compile(
    r"\b(20\d{2})\s*·\s*(Kindergarten\s*1|Foundation|Year\s*[1-4])"
    r"(?:\s*·\s*(Semester\s*(?:One|Two|1|2)|FINAL ASSESSMENT REPORT))?",
    flags=re.I,
)
YEAR_RANGE_PATTERN = re.compile(r"\b(20\d{2}\s*[–-]\s*20\d{2})\b")


def parse_insights(text: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {"title": "Learning Journey", "preface": [], "sections": {}}
    current_section: dict[str, Any] | None = None
    current_item: dict[str, Any] | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        heading = SECTION_PATTERN.match(stripped)
        if heading:
            current_section = {
                "number": int(heading.group(1)),
                "title": heading.group(2),
                "items": [],
            }
            parsed["sections"][int(heading.group(1))] = current_section
            current_item = None
            continue
        if current_section is None:
            if parsed["title"] == "Learning Journey" and not stripped.startswith("-"):
                parsed["title"] = stripped
            else:
                parsed["preface"].append(stripped.removeprefix("- "))
            continue
        if line.startswith("- "):
            current_item = {"summary": stripped[2:].strip(), "evidence": []}
            current_section["items"].append(current_item)
            continue
        if line.startswith("  -") or line.startswith("\t-"):
            if current_item is not None:
                current_item["evidence"].append(stripped[1:].strip())
            continue
        if current_item is not None:
            target = current_item["evidence"] if current_item["evidence"] else None
            if target:
                target[-1] += " " + stripped
            else:
                current_item["summary"] += " " + stripped

    for section in parsed["sections"].values():
        for item in section["items"]:
            periods = []
            for evidence in item["evidence"]:
                for year, grade, reporting_period in PERIOD_PATTERN.findall(evidence):
                    normalized = " · ".join(
                        part.strip() for part in (year, grade, reporting_period) if part.strip()
                    )
                    if normalized not in periods:
                        periods.append(normalized)
                for year_range in YEAR_RANGE_PATTERN.findall(evidence):
                    normalized = re.sub(r"\s*[–-]\s*", "–", year_range)
                    if normalized not in periods:
                        periods.append(normalized)
            item["periods"] = periods
    return parsed


def concise_story(parsed: dict[str, Any]) -> str:
    """Create a short recap using cached top-level statements verbatim."""
    strengths = parsed["sections"].get(1, {}).get("items", [])[:2]
    growth = parsed["sections"].get(2, {}).get("items", [])[:1]
    statements = [item["summary"].rstrip(".") for item in strengths + growth]
    if not statements:
        return "The cached analysis does not contain enough structured evidence for a short recap."
    return "Across the journey, the reports highlight " + "; ".join(statements) + "."
