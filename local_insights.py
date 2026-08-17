"""Local-only staged learning-insight generation through Ollama.

This module accepts an already-loaded, de-identified dataset. It never locates,
opens, or processes source PDF files.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "qwen3:4b"
CACHE_VERSION = 2
ProgressCallback = Callable[[int, int, str], None]


def dataset_hash(dataset: dict[str, Any]) -> str:
    encoded = json.dumps(dataset, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_cached_insights(cache_path: Path, dataset: dict[str, Any]) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        cached.get("cache_version") != CACHE_VERSION
        or cached.get("model") != MODEL
        or cached.get("dataset_sha256") != dataset_hash(dataset)
    ):
        return None
    return cached


def ollama_status() -> tuple[bool, str]:
    try:
        request = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(request, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = {model.get("name") for model in payload.get("models", [])}
        if MODEL not in models:
            return False, f"The local model {MODEL} is not installed."
        return True, f"Local Ollama is ready with {MODEL}."
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return False, "The local Ollama service is not reachable."


def _call_ollama(system: str, prompt: str, max_tokens: int) -> str:
    # This qwen3:4b package's chat template unconditionally opens a <think>
    # block. Raw mode preserves the same local chat format while starting the
    # assistant directly inside a final-answer envelope, so reasoning is not
    # generated or exposed.
    clean_system = system.replace("<|", "< |")
    clean_prompt = prompt.replace("<|", "< |")
    raw_prompt = (
        f"<|im_start|>system\n{clean_system}\n"
        "Return one JSON object with a single content string containing only the requested user-visible answer. "
        "Do not include analysis or chain-of-thought."
        "<|im_end|>\n"
        f"<|im_start|>user\n{clean_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    payload = {
        "model": MODEL,
        "stream": False,
        "think": False,
        "raw": True,
        "prompt": raw_prompt,
        "format": {
            "type": "object",
            "properties": {"content": {"type": "string"}},
            "required": ["content"],
            "additionalProperties": False,
        },
        "options": {
            "temperature": 0.15,
            "num_ctx": 8192,
            "num_predict": max_tokens,
        },
    }
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Local Ollama returned HTTP {error.code}: {detail[:300]}") from error
    except (OSError, urllib.error.URLError) as error:
        raise RuntimeError("The local Ollama request failed or timed out.") from error
    try:
        structured = json.loads(result.get("response", ""))
        final_content = structured["content"].strip()
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as error:
        raise RuntimeError("The local model did not return safe structured output; nothing was cached.") from error
    if not final_content:
        raise RuntimeError("The local model returned an empty response.")
    return final_content


def _period_label(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    parts = (metadata.get("year"), metadata.get("grade"), metadata.get("reporting_period"))
    return " · ".join(str(part) for part in parts if part)


def _unique_text_sections(report: dict[str, Any]) -> list[dict[str, str]]:
    priority = {
        "learning_goal_or_next_step": 0,
        "teacher_comment_or_observation": 1,
        "learning_disposition_or_personal_indicator": 2,
        "learning_content": 3,
        "assessment_context": 4,
    }
    seen: set[str] = set()
    sections = []
    for section in sorted(
        report["text_sections"],
        key=lambda item: (priority.get(item["type"], 9), item["page"], item["bbox"][1]),
    ):
        text = " ".join(section["text"].split())
        if len(text) < 25 or text in seen:
            continue
        seen.add(text)
        sections.append({"type": section["type"], "text": text})
    return sections


def _compact_ratings(report: dict[str, Any]) -> list[list[Any]]:
    ratings = []
    for rating in report.get("graphical_ratings", []):
        if not rating.get("assessment_area"):
            continue
        ratings.append(
            [
                rating["assessment_area"],
                rating.get("number_of_positions"),
                rating.get("previous_position"),
                rating.get("current_position"),
                rating.get("reference_position"),
                rating.get("curriculum_level"),
            ]
        )
    return ratings


def compact_longitudinal_evidence(
    dataset: dict[str, Any], character_limit: int = 24000
) -> dict[str, Any]:
    """Build one bounded longitudinal packet directly from de-identified JSON."""
    reports = dataset.get("reports", [])
    packet: dict[str, Any] = {
        "scope": "de-identified chronological learning evidence",
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
    # Reserve a fair text budget for every reporting period before filling it.
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
        for section in _unique_text_sections(report):
            section_type = section["type"]
            if section_type == "learning_goal_or_next_step":
                destination = "goals_and_next_steps"
            elif section_type == "learning_disposition_or_personal_indicator":
                destination = "learning_behaviours"
            elif section_type in {"teacher_comment_or_observation", "learning_content"}:
                destination = "teacher_observations"
            else:
                continue
            text = section["text"]
            if len(text) < 40:
                continue
            remaining = per_period_text_budget - used_text
            if remaining < 80:
                break
            if len(text) > remaining:
                text = text[: max(0, remaining - 1)].rstrip() + "…"
            period[destination].append(text)
            used_text += len(text)
        packet["reporting_periods"].append(period)

    # If rating-heavy reports exceed the target, retain all evidence but make
    # the fact explicit. No content is silently interpreted or fabricated.
    packet["package_characters"] = len(json.dumps(packet, ensure_ascii=False))
    packet["package_bounded_target"] = character_limit
    return packet


SYNTHESIS_SYSTEM = """You create a strengths-based longitudinal learning-journey synthesis from
one compact, de-identified structured evidence package. Use only supplied evidence. Do not diagnose, rank,
label, speculate, or invent. Do not normalize incompatible school rating scales. Clearly separate
school-reported marker movement from broader themes. Every significant bullet must cite at least
two supporting reporting periods when claiming a recurring or longitudinal pattern; otherwise
describe it as period-specific. Do not reveal chain-of-thought or reasoning."""


def generate_insights(
    dataset: dict[str, Any], cache_path: Path, progress: ProgressCallback
) -> dict[str, Any]:
    reports = dataset.get("reports", [])
    if not reports:
        raise ValueError("The structured dataset contains no reports.")
    ready, message = ollama_status()
    if not ready:
        raise RuntimeError(message)

    total_steps = 2
    progress(0, total_steps, "Preparing one compact longitudinal evidence package")
    packet = compact_longitudinal_evidence(dataset)
    progress(1, total_steps, "Generating the longitudinal insights locally")
    synthesis_prompt = """Create polished Markdown with these exact sections:
## Recurring strengths
## Areas of growth
## How learning happens
## Academic journey themes
## Earlier goals connected to later evidence
## Themes in teacher observations
## Other evidence-supported patterns
## Important context and limits

Use positive, clear language suitable for a 10-year-old and parent to read together. Every
significant insight must cite the supporting reporting periods. Recurring or longitudinal claims
must cite multiple periods. Say plainly when a requested connection is not genuinely supported.
Do not include scores, rankings, diagnoses, predictions, speculation, or invented conclusions.
Treat academic position scales as local to their original period and distinguish reported marker
movement from broader themes. Compact longitudinal evidence package:
""" + json.dumps(packet, ensure_ascii=False)
    synthesis = _call_ollama(SYNTHESIS_SYSTEM, synthesis_prompt, max_tokens=1800)
    progress(total_steps, total_steps, "Learning Journey insights are ready")

    result = {
        "cache_version": CACHE_VERSION,
        "model": MODEL,
        "dataset_sha256": dataset_hash(dataset),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "method": "one compact Python-built evidence package and one local Ollama synthesis call",
        "evidence_period_count": len(reports),
        "evidence_package_characters": packet["package_characters"],
        "synthesis": synthesis,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(cache_path)
    return result
