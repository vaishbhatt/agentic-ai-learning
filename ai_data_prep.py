"""Prepare and audit a local, de-identified payload for possible future AI use.

This module only accepts the structured learning-journey dictionary. It has no
network code and never locates, opens, or processes source PDF files.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

REMOVAL_PATTERNS = (
    ("email_address", re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "[REMOVED EMAIL]"),
    ("phone_number", re.compile(r"\b(?:\+?61|0)[\d ()-]{7,}\b"), "[REMOVED PHONE]"),
    (
        "student_identifier",
        re.compile(r"(?i)\b(?:student\s*(?:id|number)|registration\s*(?:id|number))\s*[:#-]?\s*[A-Z0-9-]+"),
        "[REMOVED STUDENT ID]",
    ),
    (
        "date_of_birth",
        re.compile(r"(?i)\b(?:date\s+of\s+birth|dob)\s*[:#-]?\s*\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}"),
        "[REMOVED DOB]",
    ),
    (
        "address",
        re.compile(r"(?i)\baddress\s*:\s*[^\n|;]+"),
        "[REMOVED ADDRESS]",
    ),
    (
        "staff_name",
        re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Dr)\.?\s+[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+)?"),
        "[REMOVED STAFF NAME]",
    ),
    (
        "school_name",
        re.compile(
            r"\b[A-Z][A-Za-z'&.-]*(?:\s+[A-Z][A-Za-z'&.-]*){0,6}\s+"
            r"(?:Primary\s+School|School|College|Academy|Institute|University|Learning\s+Centre)\b"
        ),
        "[REMOVED ORGANISATION]",
    ),
    (
        "labelled_organisation_or_location",
        re.compile(
            r"(?i)\b(?:organisation|organization|school|campus|centre|center|location)\s*:\s*[^\n|;]+"
        ),
        "[REMOVED ORGANISATION OR LOCATION]",
    ),
    (
        "street_address",
        re.compile(
            r"\b\d{1,5}\s+[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,4}\s+"
            r"(?:Street|St|Road|Rd|Avenue|Ave|Drive|Dr|Lane|Ln|Court|Ct|Place|Pl|Boulevard|Blvd)\b\.?",
            flags=re.I,
        ),
        "[REMOVED LOCATION]",
    ),
    (
        "possible_person_name",
        re.compile(r"\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b"),
        "[REMOVED POSSIBLE PERSON NAME]",
    ),
)


def _period_label(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    return " · ".join(
        str(part)
        for part in (metadata.get("year"), metadata.get("grade"), metadata.get("reporting_period"))
        if part
    )


def _unique_learning_sections(report: dict[str, Any]) -> list[dict[str, str]]:
    priority = {
        "learning_goal_or_next_step": 0,
        "teacher_comment_or_observation": 1,
        "learning_disposition_or_personal_indicator": 2,
        "learning_content": 3,
    }
    seen: set[str] = set()
    sections = []
    for section in sorted(
        report.get("text_sections", []),
        key=lambda item: (priority.get(item["type"], 9), item["page"], item["bbox"][1]),
    ):
        text = " ".join(section["text"].split())
        if section["type"] not in priority or len(text) < 25 or text in seen:
            continue
        seen.add(text)
        sections.append({"type": section["type"], "text": text})
    return sections


def _compact_ratings(report: dict[str, Any]) -> list[list[Any]]:
    return [
        [
            rating["assessment_area"],
            rating.get("number_of_positions"),
            rating.get("previous_position"),
            rating.get("current_position"),
            rating.get("reference_position"),
            rating.get("curriculum_level"),
        ]
        for rating in report.get("graphical_ratings", [])
        if rating.get("assessment_area")
    ]


def compact_longitudinal_evidence(
    dataset: dict[str, Any], character_limit: int = 24000
) -> dict[str, Any]:
    reports = dataset.get("reports", [])
    packet: dict[str, Any] = {
        "important_interpretation_rule": (
            "Position scales are meaningful only within their original reporting period; "
            "do not equate different schools or rating systems."
        ),
        "academic_rating_fields": [
            "area",
            "scale_positions",
            "previous",
            "current",
            "reference",
            "curriculum_label",
        ],
        "reporting_periods": [],
    }
    per_period_text_budget = max(900, (character_limit - 12000) // max(1, len(reports)))
    for report in reports:
        period: dict[str, Any] = {
            "reporting_period": _period_label(report),
            "academic_ratings": _compact_ratings(report),
            "learning_behaviours": [],
            "teacher_observations": [],
            "goals_and_next_steps": [],
        }
        used_text = 0
        for section in _unique_learning_sections(report):
            if section["type"] == "learning_goal_or_next_step":
                destination = "goals_and_next_steps"
            elif section["type"] == "learning_disposition_or_personal_indicator":
                destination = "learning_behaviours"
            else:
                destination = "teacher_observations"
            text = section["text"]
            remaining = per_period_text_budget - used_text
            if remaining < 80:
                break
            if len(text) > remaining:
                text = text[: max(0, remaining - 1)].rstrip() + "…"
            period[destination].append(text)
            used_text += len(text)
        packet["reporting_periods"].append(period)
    return packet

SCAN_PATTERNS = (
    ("possible_email", re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "high"),
    ("possible_phone", re.compile(r"\b(?:\+?61|0)[\d ()-]{7,}\b"), "high"),
    ("possible_identity_label", re.compile(r"(?i)\b(?:student\s*(?:name|id|number)|dob|date\s+of\s+birth|address)\b"), "high"),
    ("possible_school_name", re.compile(r"(?i)\b(?:primary\s+school|college|academy)\b"), "high"),
    ("possible_staff_name", re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Dr)\.?\s+[A-Z]"), "high"),
    ("possible_person_name", re.compile(r"\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b"), "high"),
    (
        "possible_street_address",
        re.compile(
            r"\b\d{1,5}\s+[A-Za-z.'-]+(?:\s+[A-Za-z.'-]+){0,4}\s+"
            r"(?:Street|St|Road|Rd|Avenue|Ave|Drive|Dr|Lane|Ln|Court|Ct|Place|Pl|Boulevard|Blvd)\b\.?",
            flags=re.I,
        ),
        "high",
    ),
    (
        "possible_organisation_or_location_label",
        re.compile(r"(?i)\b(?:organisation|organization|school|campus|centre|center|location|address)\s*:"),
        "high",
    ),
)


def _sanitize_text(value: str, removals: Counter[str]) -> str:
    placeholders = (
        ("student_name_placeholder", "[STUDENT]", "the learner"),
        ("staff_name_placeholder", "[STAFF]", "the teacher"),
        ("school_name_placeholder", "[SCHOOL]", "the school"),
        ("generic_redaction_placeholder", "[REDACTED]", ""),
    )
    for category, marker, replacement in placeholders:
        removals[category] += value.count(marker)
        value = value.replace(marker, replacement)
    for category, pattern, replacement in REMOVAL_PATTERNS:
        value, count = pattern.subn(replacement, value)
        removals[category] += count
    return " ".join(value.split())


def _sanitize(value: Any, removals: Counter[str]) -> Any:
    if isinstance(value, str):
        return _sanitize_text(value, removals)
    if isinstance(value, list):
        return [_sanitize(item, removals) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize(item, removals) for key, item in value.items()}
    return value


def _walk_strings(value: Any, path: str = "$") -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_strings(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_strings(item, f"{path}.{key}")


def privacy_scan(payload: dict[str, Any]) -> dict[str, Any]:
    findings = []
    for path, value in _walk_strings(payload):
        for category, pattern, severity in SCAN_PATTERNS:
            matches = list(pattern.finditer(value))
            if matches:
                findings.append(
                    {
                        "category": category,
                        "severity": severity,
                        "json_path": path,
                        "match_count": len(matches),
                    }
                )
    high_risk = sum(item["match_count"] for item in findings if item["severity"] == "high")
    review = sum(item["match_count"] for item in findings if item["severity"] == "review")
    return {
        "status": "blocked" if high_risk else ("review_recommended" if review else "clear"),
        "high_risk_match_count": high_risk,
        "review_match_count": review,
        "findings": findings,
        "note": "Findings contain categories and JSON paths only; matched text is never copied into this report.",
    }


def prepare_ai_payload(dataset: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    compact = compact_longitudinal_evidence(dataset)
    # Keep only learning evidence and the one essential interpretation rule.
    candidate = {
        "interpretation_rule": compact["important_interpretation_rule"],
        "academic_rating_fields": compact["academic_rating_fields"],
        "reporting_periods": compact["reporting_periods"],
    }
    removals: Counter[str] = Counter()
    payload = _sanitize(candidate, removals)
    report = privacy_scan(payload)
    report["removed_counts"] = dict(sorted(removals.items()))
    report["payload_character_count"] = len(payload_json(payload))
    report["payload_word_count"] = len(re.findall(r"\b\w+\b", payload_json(payload)))
    report["source"] = "learning_journey_output/learning_journey.json only"
    report["network_activity"] = "none"
    return payload, report


def payload_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def save_prepared_payload(
    dataset: dict[str, Any], payload_path: Path, report_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, report = prepare_ai_payload(dataset)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(payload_json(payload), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload, report
