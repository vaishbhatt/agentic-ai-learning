"""One-request OpenAI Responses API integration for sanitized learning evidence."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


MODEL = "gpt-5-mini"
RESPONSES_URL = "https://api.openai.com/v1/responses"
CACHE_VERSION = 2


def api_key_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def payload_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_cached_insights(cache_path: Path, payload: dict[str, Any]) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        cached.get("cache_version") != CACHE_VERSION
        or cached.get("model") != MODEL
        or cached.get("payload_sha256") != payload_hash(payload)
    ):
        return None
    return cached


def _privacy_gate(privacy_report: dict[str, Any]) -> None:
    if (
        privacy_report.get("status") != "clear"
        or privacy_report.get("high_risk_match_count", 0) != 0
        or privacy_report.get("review_match_count", 0) != 0
    ):
        raise ValueError(
            "The local privacy scan is not Clear. Review and sanitize every flagged JSON path "
            "before any OpenAI API request is allowed."
        )


def _extract_output_text(response: dict[str, Any]) -> str:
    chunks = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                chunks.append(content["text"])
    text = "\n".join(chunks).strip()
    if not text:
        raise RuntimeError("OpenAI returned no displayable insight text.")
    return text


def generate_openai_insights(
    payload: dict[str, Any],
    privacy_report: dict[str, Any],
    cache_path: Path,
    *,
    transport: Callable[..., Any] = urllib.request.urlopen,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Send exactly one sanitized payload after a mandatory local privacy gate."""
    _privacy_gate(privacy_report)
    key = (api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")).strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set in the application environment.")

    instructions = """Create a warm, strengths-based longitudinal Learning Journey for a
10-year-old and their parent. Use only the sanitized structured evidence supplied. Cover recurring
strengths, meaningful growth, learning-behaviour patterns, academic themes, recurring teacher
observations, and connections between earlier goals and later evidence only when genuinely
supported. Every significant conclusion must cite its supporting reporting periods and evidence.
Treat each school's scale only within its original reporting period. Do not diagnose, rank, label,
speculate, predict, infer identities, or invent conclusions. If evidence is insufficient, say so.
Return exactly seven plain numbered section headings in this order:
1) Recurring strengths (what the reports consistently highlight)
2) Meaningful growth over time (evidence of development)
3) Learning-behaviour patterns (recurring classroom behaviours and social learning)
4) Academic themes and consistency across subjects
5) Recurring teacher observations (phrases that appear across reports)
6) Connections between earlier goals and later evidence (only where directly supported)
7) Areas where evidence is insufficient to draw conclusions
Under each heading, use concise top-level '- ' insight bullets. Put detailed support beneath each
insight as indented '  - ' evidence bullets beginning with the reporting period. Do not add other
headings. This stable structure is required by the local visual presentation."""
    request_body = {
        "model": MODEL,
        "instructions": instructions,
        "input": (
            "Sanitized longitudinal learning evidence follows. This is the complete evidence; "
            "do not request or infer source documents.\n\n"
            + json.dumps(payload, ensure_ascii=False)
        ),
        "max_output_tokens": 2200,
        "reasoning": {"effort": "low"},
    }
    request = urllib.request.Request(
        RESPONSES_URL,
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with transport(request, timeout=240) as response:
            api_response = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        # Never include request headers, payload, or the API key in errors/cache.
        raise RuntimeError(f"OpenAI API request failed with HTTP {error.code}.") from error
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise RuntimeError("The OpenAI API request failed or returned invalid data.") from error

    result = {
        "cache_version": CACHE_VERSION,
        "model": MODEL,
        "payload_sha256": payload_hash(payload),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "insights": _extract_output_text(api_response),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(cache_path)
    return result
