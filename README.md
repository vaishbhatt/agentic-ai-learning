# Little Chapters

Little Chapters is a privacy-first Streamlit application that brings school
reports together as one chronological, strengths-focused learning story for a
parent and child to explore together. Every report is a chapter. See the whole
learning story.

The public repository opens immediately in **Demo Mode** using entirely
synthetic data. Users can also upload their own PDF reports for local,
deterministic extraction.

## What the application includes

- **My Journey** — a chronological reporting-period timeline.
- **Academic Journey** — report-specific achievement and progression markers.
- **How I Learn** — effort, behaviour and approaches-to-learning evidence in
  each report's original terminology.
- **Through My Teachers' Eyes** — observations, comments, goals and next steps.
- **AI Learning Insights** — optional longitudinal semantic analysis after a
  local privacy scan and explicit user consent.

Different schools' rating systems are never converted into a common score.
Uncertain values remain unset and are flagged rather than guessed.

## Architecture

```text
PDF Reports
    ↓
Local deterministic extraction
    ↓
Structured Learning Journey
    ↓
Dashboard
    ↓
Local privacy sanitisation
    ↓
Optional OpenAI analysis
```

The original PDF bytes stop at the deterministic extraction stage. They are
never included in an AI payload.

## Installation

Python 3.12 is required. With [uv](https://docs.astral.sh/uv/):

```powershell
uv sync
uv run streamlit run app.py
```

Or with a conventional virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

## Explore Demo Journey

Choose **Explore Demo Journey** when the app opens. The dashboard loads
[`demo_data/learning_journey_demo.json`](demo_data/learning_journey_demo.json),
an independently written fictional journey from Kindergarten to Year 4.

The demo contains no real learner, school, teacher or report content and works
without PDFs, an API key or internet access.

## Upload Reports

Choose **Upload Reports**, select one or more PDFs, then choose **Extract
Uploaded Reports**.

```text
Upload PDFs → Local extraction → Structured Learning Journey → Dashboard
```

Uploaded PDF bytes are processed in memory and are not permanently saved by
the app. The resulting structured journey remains in the current Streamlit
session. A validation summary reports the pages, text sections, tables,
graphical ratings and limitations found for each upload. The app limits each
selection to 75 MB in total to avoid excessive memory use.

### Deterministic extraction

The extractor uses PyMuPDF—not an LLM—to:

- read native PDF text blocks and their layout;
- detect tables supported by PDF text/line geometry;
- preserve narrative comments and report terminology;
- redact common identity fields;
- detect a known family of graphical assessment scales from vector-cell and
  rendered-marker geometry;
- sort reports using confidently extracted year, grade and reporting period.

An LLM is deliberately not used here because extraction should be repeatable,
inspectable and incapable of inventing missing ratings.

### Currently recognised report structures

The graphical detector was developed around reports with:

- rows of 8–12 evenly spaced vector cells;
- a filled dark circle for the current position;
- an optional hollow circle and dashed connector for the previous position;
- a non-white filled cell for an expected/reference position; and
- nearby curriculum headings formatted like `Level 2`.

Metadata detection recognises common `Foundation`, `Prep`, `Year/Grade 1–4`,
semester and term labels. These are useful patterns, not a claim that every
school-report format is supported.

For unknown layouts the app extracts reliable native text and supported tables,
allows partial results, and flags missing or ambiguous fields. It does not use
OCR, guess graphical positions or infer curriculum mappings. Image-only scans
may therefore yield little usable content.

## Structured data and the dashboard

Both demo and uploaded journeys use the same JSON-compatible schema:

- report metadata and reporting periods;
- typed learning-text sections;
- structured table rows;
- graphical assessment rows with previous/current/reference positions; and
- extraction flags and confidence information.

All five dashboard views consume this shared model; there is no separate demo
dashboard implementation.

## Optional AI insights

The deterministic dashboard works fully without an API key. To enable the
optional OpenAI insight button, set the key only in the process environment:

```powershell
$env:OPENAI_API_KEY = "your-key"
uv run streamlit run app.py
```

Never commit the key or place it in the project. The application reads
`OPENAI_API_KEY` at request time and does not print, store or cache it.

Before an API request is enabled, Python creates a compact evidence payload and
runs a conservative local privacy scan. The app shows **Preview AI Data** so the
user can inspect the exact payload. Generation is blocked unless the scan is
`Clear`, and a request occurs only after an explicit **Generate AI Insights**
click. Only the sanitized structured evidence is sent to OpenAI in one Responses
API request; original PDFs and the unsanitized dataset are never sent.

## Privacy by design

- Demo data is fully synthetic.
- Uploaded PDFs are processed locally in memory.
- No OCR, cloud extraction service or external PDF upload is used.
- Names, schools, staff, IDs, DOBs, addresses and contact details are targeted
  by deterministic redaction and a final conservative privacy scan.
- AI is optional and separated from extraction.
- Private source and output directories are excluded by `.gitignore`.
- Streamlit usage-stat collection is disabled in `.streamlit/config.toml`.

## Important limitation

Automated privacy and PDF-structure detection cannot guarantee support for
every document. Review the extraction summary and exact AI payload before using
results or enabling optional external analysis.
