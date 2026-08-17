"""Local Learning Journey dashboard backed by the extracted JSON dataset."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import streamlit as st

from ai_data_prep import payload_json, prepare_ai_payload
from insight_presentation import concise_story, parse_insights
from openai_insights import (
    MODEL as OPENAI_MODEL,
    api_key_available,
    generate_openai_insights,
    load_cached_insights,
    payload_hash,
)
from uploaded_reports import FORMAT_LIMITATIONS, extract_uploaded_reports


DEMO_DATA_PATH = Path(__file__).parent / "demo_data" / "learning_journey_demo.json"
DEMO_INSIGHTS_PATH = Path(__file__).parent / "demo_data" / "ai_insights_demo.json"
APP_CACHE_DIR = Path(__file__).parent / ".app_cache"

st.set_page_config(page_title="Little Chapters", page_icon="🌱", layout="wide")


@st.cache_data
def load_data(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe(value: Any) -> str:
    return html.escape(str(value or ""))


def growth_motif(stage: str = "plant", css_class: str = "growth-motif") -> str:
    """Original inline SVG used as the product's learning-and-growth motif."""
    stages = {
        "seed": "<ellipse cx='32' cy='48' rx='7' ry='4'/>",
        "sprout": (
            "<path d='M32 49 C32 39 31 31 32 23'/>"
            "<path d='M32 31 C23 30 18 25 18 18 C26 18 31 22 32 31Z'/>"
            "<path d='M32 25 C38 24 43 20 44 14 C37 14 33 18 32 25Z'/>"
        ),
        "plant": (
            "<path class='plant-stem' d='M32 51 C31 38 32 25 34 12'/>"
            "<path class='plant-leaf leaf-left' d='M32 39 C21 38 15 32 14 23 C24 22 31 28 32 39Z'/>"
            "<path class='plant-leaf leaf-right' d='M33 29 C43 28 49 22 50 14 C41 14 35 20 33 29Z'/>"
            "<path class='plant-root' d='M32 47 C25 47 21 44 19 39'/><path class='plant-root' d='M32 47 C39 47 43 44 45 39'/>"
        ),
    }
    drawing = stages.get(stage, stages["plant"])
    return (
        f"<svg class='{safe(css_class)}' viewBox='0 0 64 64' role='img' aria-label='Learning growth'>"
        f"<g fill='none' stroke='currentColor' stroke-width='3.2' stroke-linecap='round' "
        f"stroke-linejoin='round'>{drawing}</g></svg>"
    )


def period_label(report: dict[str, Any]) -> str:
    meta = report["metadata"]
    return " · ".join(str(value) for value in (meta.get("grade"), meta.get("reporting_period"), meta.get("year")) if value)


def subject_group(label: str | None) -> str:
    """Navigation grouping only; never used to convert or compare ratings."""
    value = (label or "Other assessment area").lower()
    groups = (
        ("English", ("reading", "writing", "speaking", "english")),
        ("Mathematics", ("mathematics", "number", "algebra", "geometry", "statistics")),
        ("Capabilities", ("personal", "social", "ethical", "intercultural")),
        ("Health & Physical Education", ("physical", "health")),
        ("The Arts", ("arts", "music")),
        ("Humanities", ("humanities", "history", "geography", "civics")),
        ("Science", ("science",)),
        ("Technologies", ("technolog",)),
        ("Languages", ("indonesian",)),
    )
    return next((group for group, terms in groups if any(term in value for term in terms)), "Other assessment areas")


def sections_of(report: dict[str, Any], kinds: set[str]) -> list[dict[str, Any]]:
    return [section for section in report["text_sections"] if section["type"] in kinds]


def narrative_sections(report: dict[str, Any]) -> list[dict[str, Any]]:
    explicit = sections_of(report, {"teacher_comment_or_observation", "learning_goal_or_next_step"})
    prose = [
        section for section in report["text_sections"]
        if section["type"] == "learning_content" and len(section["text"]) >= 120
    ]
    return explicit + prose


def render_upload_workflow() -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    st.markdown(
        "<div class='source-banner uploaded'><strong>Uploaded Journey</strong> — PDFs are processed "
        "locally in memory. They are not saved by the app or sent to an AI service.</div>",
        unsafe_allow_html=True,
    )
    uploads = st.file_uploader(
        "Upload one or more PDF school reports",
        type=["pdf"],
        accept_multiple_files=True,
        help="Native-text PDFs work best. Scanned pages are not OCR-processed.",
    )
    total_upload_bytes = sum(uploaded.size for uploaded in uploads) if uploads else 0
    upload_too_large = total_upload_bytes > 75 * 1024 * 1024
    if upload_too_large:
        st.error("The selected PDFs exceed the 75 MB total in-memory processing limit.")
    process = st.button(
        "Extract Uploaded Reports",
        type="primary",
        disabled=not uploads or upload_too_large,
        key="extract_uploaded_reports",
    )
    if process and uploads:
        with st.spinner("Extracting text, tables and recognised graphical scales locally…"):
            try:
                pairs = [(uploaded.name, uploaded.getvalue()) for uploaded in uploads]
                dataset, summaries = extract_uploaded_reports(pairs)
                st.session_state["uploaded_journey_data"] = dataset
                st.session_state["uploaded_journey_summaries"] = summaries
            except Exception as error:
                st.error(str(error))

    dataset = st.session_state.get("uploaded_journey_data")
    summaries = st.session_state.get("uploaded_journey_summaries", [])
    if dataset:
        st.success(
            f"Uploaded Journey ready: {dataset['report_count']} report(s) produced structured learning data."
        )
        with st.expander("Extraction and validation summary", expanded=True):
            for summary in summaries:
                st.markdown(
                    f"**Upload {summary['upload']} — {summary['status'].replace('_', ' ').title()}**  "
                    f"  \nPages: {summary['pages']} · Text sections: {summary['text_sections']} · "
                    f"Tables: {summary['tables']} · Graphical ratings: {summary['graphical_ratings']}"
                )
                for flag in summary["flags"]:
                    st.warning(flag)
            st.markdown("**Known format limitations**")
            for limitation in FORMAT_LIMITATIONS:
                st.markdown(f"- {limitation}")
    else:
        st.info("Upload PDFs and choose Extract Uploaded Reports to create an Uploaded Journey.")
    return dataset, summaries


def render_overview(reports: list[dict[str, Any]]) -> None:
    years = {r["metadata"].get("year") for r in reports if r["metadata"].get("year")}
    areas = {x.get("assessment_area") for r in reports for x in r["graphical_ratings"] if x.get("assessment_area")}
    goals = sum(s["type"] == "learning_goal_or_next_step" for r in reports for s in r["text_sections"])
    metrics = (
        ("Reporting periods", len(reports), "Kinder to Year 4"),
        ("Years represented", len(years), "chronological records"),
        ("Charted areas", len(areas), "original report labels"),
        ("Goals & next steps", goals, "recorded by educators"),
    )
    for column, (label, value, caption) in zip(st.columns(4), metrics):
        with column:
            st.markdown(f"<div class='metric-card'><span>{safe(label)}</span><strong>{value}</strong><small>{safe(caption)}</small></div>", unsafe_allow_html=True)


def render_timeline(reports: list[dict[str, Any]], selected: int) -> None:
    items = []
    for index, report in enumerate(reports):
        meta = report["metadata"]
        active = " active" if index == selected else ""
        progress = index / max(1, len(reports) - 1)
        stage = "seed" if progress < .25 else "sprout" if progress < .7 else "plant"
        items.append(
            f"<div class='timeline-item{active}'><div class='timeline-dot'></div>"
            f"<div class='timeline-growth'>{growth_motif(stage, 'growth-motif small')}</div>"
            f"<div class='timeline-year'>{safe(meta.get('year') or 'Year not recorded')}</div>"
            f"<div class='timeline-grade'>{safe(meta.get('grade') or 'Stage not recorded')}</div>"
            f"<div class='timeline-period'>{safe(meta.get('reporting_period') or 'Period not recorded')}</div></div>"
        )
    st.markdown(f"<div class='timeline'>{''.join(items)}</div>", unsafe_allow_html=True)


def rating_track(rating: dict[str, Any]) -> str:
    cells = []
    for position in range(1, int(rating["number_of_positions"]) + 1):
        classes = "track-cell reference" if position == rating.get("reference_position") else "track-cell"
        markers = ""
        if position == rating.get("previous_position"):
            markers += "<span class='previous-marker' title='Previous reported position'></span>"
        if position == rating.get("current_position"):
            markers += "<span class='current-marker' title='Current reported position'></span>"
        cells.append(f"<div class='{classes}'>{markers}</div>")
    return f"<div class='rating-track'>{''.join(cells)}</div>"


def rating_interpretation(rating: dict[str, Any]) -> tuple[str, str, list[str]]:
    """Explain markers within one reported scale without comparing report systems."""
    previous = rating.get("previous_position")
    current = rating.get("current_position")
    reference = rating.get("reference_position")
    curriculum = rating.get("curriculum_level")
    badges = [str(curriculum)] if curriculum else []

    if current is None:
        return "No current position shown", "This report does not provide a confident current marker for this area.", badges

    statements = []
    if previous is not None:
        change = current - previous
        if change > 0:
            statements.append(f"The current marker is {change} step{'s' if change != 1 else ''} ahead of the previous marker")
            badges.append("Progress since previous")
        elif change == 0:
            statements.append("The current marker is in the same position as the previous marker")
            badges.append("Same as previous")
        else:
            distance = abs(change)
            statements.append(f"The current marker is {distance} step{'s' if distance != 1 else ''} behind the previous marker")
            badges.append("Behind previous marker")

    if reference is not None:
        if current > reference:
            statements.append("above the reference position")
            badges.append("Above reference")
        elif current == reference:
            statements.append("at the reference position")
            badges.append("At reference")
        else:
            statements.append("below the reference position")
            badges.append("Below reference")

    if previous is not None and current > previous:
        headline = "Progressed since the previous assessment"
    elif previous is not None and current == previous:
        headline = "Position remained steady"
    elif previous is not None:
        headline = "Marker changed since the previous assessment"
    elif reference is not None:
        headline = "Current position shown against the reference"
    else:
        headline = "Current reported position"

    if not statements:
        sentence = f"The current marker is at position {current} on this report's scale."
    elif len(statements) == 1:
        sentence = statements[0] + "."
    else:
        sentence = statements[0] + " and " + statements[1] + "."
    return headline, sentence, badges


def render_rating(rating: dict[str, Any], period: str | None = None) -> None:
    title = safe(rating.get("assessment_area") or "Assessment area not labelled")
    if period:
        title += f" <span class='period-chip'>{safe(period)}</span>"
    headline, interpretation, badges = rating_interpretation(rating)
    badge_html = "".join(f"<span>{safe(badge)}</span>" for badge in badges)
    st.markdown(
        f"<div class='rating-card'><div class='rating-title'>{title}</div>"
        f"<div class='rating-meaning'><strong>{safe(headline)}</strong><p>{safe(interpretation)}</p>"
        f"<div class='rating-badges'>{badge_html}</div></div>{rating_track(rating)}",
        unsafe_allow_html=True,
    )
    if rating.get("flags"):
        st.markdown("<div class='uncertain'>Some details were unclear in this report, so they are not interpreted here.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_section_cards(sections: list[dict[str, Any]], empty: str) -> None:
    if not sections:
        st.info(empty)
        return
    for section in sorted(sections, key=lambda item: (item["page"], item["bbox"][1])):
        kind = safe(section["type"].replace("_", " ").title())
        content = safe(section["text"]).replace("\n", "<br>")
        st.markdown(f"<div class='story-card'><div class='story-meta'>{kind} · page {section['page']}</div><div>{content}</div></div>", unsafe_allow_html=True)


def render_my_journey(reports: list[dict[str, Any]]) -> None:
    st.markdown("## My Journey")
    st.write("A chapter-by-chapter view of growing, exploring and learning.")
    labels = [period_label(report) for report in reports]
    chosen = st.selectbox("Choose a reporting period", labels, key="journey_period")
    index = labels.index(chosen)
    report = reports[index]
    render_timeline(reports, index)
    meta = report["metadata"]
    st.markdown(f"### {safe(meta.get('grade') or 'Learning stage')} · {safe(meta.get('reporting_period') or 'Reporting period')}")
    columns = st.columns(3)
    columns[0].metric("Learning notes", len(report["text_sections"]))
    columns[1].metric("Assessment rows", len(report["graphical_ratings"]))
    columns[2].metric("Structured tables", len(report["tables"]))
    snapshot, highlights = st.tabs(["Assessment snapshot", "Learning highlights"])
    with snapshot:
        if report["graphical_ratings"]:
            st.caption("Positions are shown on this report’s own scale only.")
            for rating in report["graphical_ratings"]:
                render_rating(rating)
        else:
            st.info("This report does not contain a confidently supported graphical scale.")
    with highlights:
        content = [s for s in report["text_sections"] if s["type"] in {"learning_content", "teacher_comment_or_observation"} and len(s["text"]) >= 100][:12]
        render_section_cards(content, "No narrative learning highlights were extracted for this period.")


def demo_table_indicators(report: dict[str, Any]) -> list[tuple[str, str]]:
    indicators = []
    for table in report["tables"]:
        for row in (table.get("rows") or [])[1:]:
            if len(row) >= 2 and row[0] and row[1]:
                indicators.append((str(row[0]), str(row[1])))
    return indicators


def render_demo_my_journey(reports: list[dict[str, Any]]) -> None:
    st.markdown("## ✨ My Journey")
    st.write("A chapter-by-chapter view of growing, exploring and learning.")
    labels = [period_label(report) for report in reports]
    chosen = st.selectbox("Choose a reporting period", labels, key="demo_journey_period")
    index = labels.index(chosen)
    report = reports[index]
    render_timeline(reports, index)
    meta = report["metadata"]
    st.markdown(
        f"<div class='selected-period'><small>SELECTED CHAPTER</small>"
        f"<h2>{safe(meta.get('grade'))} · {safe(meta.get('reporting_period'))} · {safe(meta.get('year'))}</h2></div>",
        unsafe_allow_html=True,
    )

    observations = sections_of(report, {"teacher_comment_or_observation"})
    approaches = sections_of(report, {"learning_disposition_or_personal_indicator"})
    highlight_source = " ".join(section["text"] for section in observations + approaches)
    highlights = [part.strip() for part in re.split(r"(?<=[.!?])\s+", highlight_source) if part.strip()][:3]
    if highlights:
        st.markdown("<div class='journey-band-title'>💡 Key Highlights This Period</div>", unsafe_allow_html=True)
        for column, highlight, icon in zip(st.columns(len(highlights)), highlights, ("♡", "◎", "↻")):
            with column:
                st.markdown(f"<div class='highlight-card'><span>{icon}</span><p>{safe(highlight)}</p></div>", unsafe_allow_html=True)

    teacher_column, learning_column = st.columns((1.35, 1))
    with teacher_column:
        st.markdown("### Teacher Observations")
        for observation in observations:
            st.markdown(f"<div class='quote-card'><span>“</span><p>{safe(observation['text'])}</p></div>", unsafe_allow_html=True)
        if not observations:
            st.info("No teacher observation is available for this period.")
    with learning_column:
        st.markdown("### Approach to Learning")
        for approach in approaches:
            st.markdown(
                f"<div class='approach-card'><small>IN THE SCHOOL'S WORDS</small><p>{safe(approach['text'])}</p></div>",
                unsafe_allow_html=True,
            )

    indicators = demo_table_indicators(report)
    if indicators:
        st.markdown("### Learning Indicators")
        columns = st.columns(min(3, len(indicators)))
        for position, (indicator, status) in enumerate(indicators):
            with columns[position % len(columns)]:
                st.markdown(
                    f"<div class='journey-indicator'><strong>{safe(indicator)}</strong><span>{safe(status)}</span></div>",
                    unsafe_allow_html=True,
                )


def render_academic_journey(reports: list[dict[str, Any]]) -> None:
    st.markdown("## Academic Journey")
    st.write("Explore school-reported curriculum markers while keeping each report’s scale in its original context.")
    all_ratings = [(r, x) for r in reports for x in r["graphical_ratings"] if x.get("assessment_area")]
    if not all_ratings:
        st.info(
            "No confidently structured graphical academic ratings are available for this journey. "
            "Review My Journey and the extraction summary for text-based evidence and format limitations."
        )
        return
    groups = sorted({subject_group(x["assessment_area"]) for _, x in all_ratings})
    group = st.selectbox("Academic area", groups, key="academic_group")
    grouped = [(r, x) for r, x in all_ratings if subject_group(x["assessment_area"]) == group]
    labels = sorted({x["assessment_area"] for _, x in grouped})
    label = st.selectbox("School-reported subject or sub-area", labels, key="academic_label")
    st.markdown("<div class='context-note'><strong>How to read this:</strong> purple shows the current position, the teal outline shows the previous position, and gold shows the school's reference position. Each card is explained only within its own report scale; different report systems should not be directly compared.</div>", unsafe_allow_html=True)
    for report, rating in grouped:
        if rating["assessment_area"] == label:
            render_rating(rating, period_label(report))
    with st.expander("Original assessment context and terminology"):
        context = [(r, s) for r in reports for s in r["text_sections"] if s["type"] == "assessment_context"]
        for report, section in context[:30]:
            st.markdown(f"**{safe(period_label(report))}** — {safe(section['text'])}")


def render_demo_academic_journey(reports: list[dict[str, Any]]) -> None:
    st.markdown("## 📚 Academic Journey")
    st.write("A year-by-year view of school-reported learning, kept within each report's own scale.")
    all_ratings = [(report, rating) for report in reports for rating in report["graphical_ratings"] if rating.get("assessment_area")]
    areas = {rating["assessment_area"] for _, rating in all_ratings}
    curriculum_labels = {rating.get("curriculum_level") for _, rating in all_ratings if rating.get("curriculum_level")}
    movements = sum(
        rating.get("previous_position") is not None
        and rating.get("current_position") != rating.get("previous_position")
        for _, rating in all_ratings
    )
    overview = st.columns(4)
    overview[0].metric("Reporting periods", len(reports))
    overview[1].metric("Academic areas", len(areas))
    overview[2].metric("Curriculum labels", len(curriculum_labels))
    overview[3].metric("Reported movements", movements)

    st.markdown("### Academic Journey at a Glance")
    groups = sorted({subject_group(rating["assessment_area"]) for _, rating in all_ratings})
    selected_group = st.selectbox("Explore an academic area", groups, key="demo_academic_group")
    st.markdown(
        "<div class='context-note'><strong>Across the journey:</strong> each card describes only the markers and curriculum wording in that reporting period. Different report systems should not be directly compared.</div>",
        unsafe_allow_html=True,
    )

    for report in reports:
        ratings = [
            rating for rating in report["graphical_ratings"]
            if rating.get("assessment_area") and subject_group(rating["assessment_area"]) == selected_group
        ]
        if not ratings:
            continue
        meta = report["metadata"]
        st.markdown(
            f"<div class='academic-period'><span>{safe(meta.get('year'))}</span>"
            f"<strong>{safe(meta.get('grade'))}</strong><small>{safe(meta.get('reporting_period'))}</small></div>",
            unsafe_allow_html=True,
        )
        columns = st.columns(min(3, len(ratings)))
        for position, rating in enumerate(ratings):
            headline, interpretation, badges = rating_interpretation(rating)
            badge_html = "".join(f"<span>{safe(badge)}</span>" for badge in badges)
            with columns[position % len(columns)]:
                st.markdown(
                    f"<div class='academic-summary-card'><small>{safe(selected_group)}</small>"
                    f"<h4>{safe(rating['assessment_area'])}</h4><strong>{safe(headline)}</strong>"
                    f"<p>{safe(interpretation)}</p><div class='rating-badges'>{badge_html}</div></div>",
                    unsafe_allow_html=True,
                )

    with st.expander("Explore individual assessment evidence"):
        st.caption("Purple is the current marker, teal is the previous marker, and gold is the school's reference position.")
        for report, rating in all_ratings:
            if subject_group(rating["assessment_area"]) == selected_group:
                render_rating(rating, period_label(report))


def render_how_i_learn(reports: list[dict[str, Any]], *, demo_mode: bool = False) -> None:
    st.markdown("## How I Learn")
    st.write("Effort, behaviour, approaches to learning and personal development—in each school’s own words.")
    available = [r for r in reports if sections_of(r, {"learning_disposition_or_personal_indicator"}) or r["tables"]]
    if not available:
        st.info(
            "No separate effort, behaviour, approaches-to-learning or supported indicator tables "
            "were confidently identified in these uploads."
        )
        return
    labels = [period_label(report) for report in available]
    chosen = st.selectbox("Reporting period", labels, key="learning_period")
    report = available[labels.index(chosen)]
    if demo_mode:
        selected_index = reports.index(report)
        render_timeline(reports, selected_index)
        meta = report["metadata"]
        st.markdown(
            f"<div class='selected-period'><small>HOW I LEARNED IN THIS CHAPTER</small>"
            f"<h2>{safe(meta.get('grade'))} · {safe(meta.get('reporting_period'))} · {safe(meta.get('year'))}</h2></div>",
            unsafe_allow_html=True,
        )
        observations = sections_of(report, {"learning_disposition_or_personal_indicator"})
        if observations:
            st.markdown(
                "<div class='learning-observation'><div class='learning-observation-icon'>🌱</div>"
                "<div><small>APPROACH TO LEARNING</small>"
                f"<p>{safe(observations[0]['text'])}</p></div></div>",
                unsafe_allow_html=True,
            )

        habits = []
        for table in report["tables"]:
            rows = table.get("rows") or []
            for row in rows[1:]:
                if len(row) >= 2 and row[0] and row[1]:
                    habits.append((str(row[0]), str(row[1])))
        if habits:
            st.markdown("### Learning Habits & Dispositions")
            columns = st.columns(min(3, len(habits)))
            icons = ("✦", "◎", "↻")
            for index, (habit, status) in enumerate(habits):
                with columns[index % len(columns)]:
                    st.markdown(
                        f"<div class='habit-card'><div class='habit-icon'>{icons[index % len(icons)]}</div>"
                        f"<strong>{safe(habit)}</strong>"
                        f"<span>{safe(status)}</span></div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.info("No separately reported learning habits are available for this period.")
        st.caption("Wording and ratings are shown exactly as reported in this fictional journey.")
        return

    render_section_cards(sections_of(report, {"learning_disposition_or_personal_indicator"}), "No separate learning-disposition narrative was identified for this period.")
    with st.expander("Original structured indicator tables"):
        if not report["tables"]:
            st.info("No structured indicator table was extracted for this report.")
        for table in report["tables"]:
            if not table.get("rows"):
                continue
            st.caption(f"Page {table['page']} · original {table['row_count']} × {table['column_count']} table")
            st.dataframe(table["rows"], hide_index=True, width="stretch")
    st.caption("Ratings and terminology are displayed as reported; no common score has been created.")


def render_demo_teachers_eyes(reports: list[dict[str, Any]]) -> None:
    st.markdown("## 💬 Through My Teachers’ Eyes")
    st.write("A chronological collection of observations, encouragement and next steps—in the original words of each report.")
    st.markdown("<div class='teacher-journey-line'></div>", unsafe_allow_html=True)

    for report in reports:
        meta = report["metadata"]
        observations = sections_of(report, {"teacher_comment_or_observation"})
        goals = sections_of(report, {"learning_goal_or_next_step"})
        if not observations and not goals:
            continue
        st.markdown(
            f"<div class='teacher-period-heading'><div class='teacher-period-dot'>✦</div>"
            f"<div><span>{safe(meta.get('year'))}</span><h3>{safe(meta.get('grade'))}</h3>"
            f"<small>{safe(meta.get('reporting_period'))}</small></div></div>",
            unsafe_allow_html=True,
        )
        observation_column, goal_column = st.columns((1.45, 1))
        with observation_column:
            if observations:
                st.markdown("<div class='teacher-card-label observation'>💬 OBSERVATION</div>", unsafe_allow_html=True)
                for observation in observations:
                    st.markdown(
                        f"<div class='teacher-quote-card'><span>“</span><p>{safe(observation['text'])}</p></div>",
                        unsafe_allow_html=True,
                    )
        with goal_column:
            if goals:
                st.markdown("<div class='teacher-card-label goal'>🎯 GOAL / NEXT STEP</div>", unsafe_allow_html=True)
                for goal in goals:
                    st.markdown(
                        f"<div class='teacher-goal-card'><p>{safe(goal['text'])}</p></div>",
                        unsafe_allow_html=True,
                    )


def render_teachers_eyes(reports: list[dict[str, Any]]) -> None:
    st.markdown("## Through My Teachers’ Eyes")
    st.write("A chronological collection of observations, encouragement and next steps—without AI interpretation.")
    labels = ["All reporting periods"] + [period_label(report) for report in reports]
    chosen = st.selectbox("Show", labels, key="teacher_period")
    selected = reports if chosen == labels[0] else [reports[labels.index(chosen) - 1]]
    for report in selected:
        comments = narrative_sections(report)
        goals = sections_of(report, {"learning_goal_or_next_step"})
        if not comments and not goals:
            continue
        with st.expander(period_label(report), expanded=len(selected) == 1):
            comment_tab, goal_tab = st.tabs(["Comments & observations", "Goals & next steps"])
            with comment_tab:
                render_section_cards(comments, "No narrative observation was extracted for this period.")
            with goal_tab:
                render_section_cards(goals, "No explicit learning goal was extracted for this period.")


def render_period_badges(periods: list[str]) -> None:
    if periods:
        st.markdown(
            "<div class='evidence-badges'>"
            + "".join(f"<span>{safe(period)}</span>" for period in periods)
            + "</div>",
            unsafe_allow_html=True,
        )


def render_evidence_expander(item: dict[str, Any]) -> None:
    evidence = item.get("evidence", [])
    if not evidence:
        return
    with st.expander(f"Supporting evidence · {len(evidence)} item{'s' if len(evidence) != 1 else ''}"):
        for detail in evidence:
            st.markdown(f"- {detail}")


def render_insight_cards(items: list[dict[str, Any]], columns: int = 3, icon: str = "✦") -> None:
    if not items:
        st.info("The cached analysis did not contain supported evidence for this section.")
        return
    card_columns = st.columns(min(columns, len(items)))
    for index, item in enumerate(items):
        with card_columns[index % len(card_columns)]:
            st.markdown(
                f"<div class='insight-card'><div class='insight-icon'>{icon}</div>"
                f"<div class='insight-summary'>{safe(item['summary'])}</div></div>",
                unsafe_allow_html=True,
            )
            render_period_badges(item.get("periods", []))
            render_evidence_expander(item)


def render_growth_timeline(items: list[dict[str, Any]]) -> None:
    if not items:
        st.info("No supported growth timeline was present in the cached analysis.")
        return
    for index, item in enumerate(items, 1):
        st.markdown(
            f"<div class='growth-row'><div class='growth-number'>{index}</div>"
            f"<div><strong>{safe(item['summary'])}</strong>"
            f"<div class='growth-path'>Kinder <span>→</span> Year 4</div></div></div>",
            unsafe_allow_html=True,
        )
        render_period_badges(item.get("periods", []))
        render_evidence_expander(item)


def render_goals_to_growth(items: list[dict[str, Any]]) -> None:
    if not items:
        st.info("The cached analysis did not identify a supported goal-to-growth connection.")
        return
    for item in items:
        summary = item["summary"]
        parts = re.split(r"\s*(?:→|->)\s*", summary, maxsplit=1)
        if len(parts) == 2:
            visual = (
                f"<div class='goal-link'><div><small>EARLIER GOAL</small><strong>{safe(parts[0])}</strong></div>"
                f"<div class='goal-arrow'>→</div><div><small>LATER EVIDENCE</small>"
                f"<strong>{safe(parts[1])}</strong></div></div>"
            )
        else:
            visual = f"<div class='goal-link single'><strong>{safe(summary)}</strong></div>"
        st.markdown(visual, unsafe_allow_html=True)
        render_period_badges(item.get("periods", []))
        render_evidence_expander(item)


def render_demo_insights_snapshot(snapshot: dict[str, Any]) -> None:
    """Present the compact overview bundled with the synthetic demo asset."""

    def snapshot_card(headline: str, sentence: str, periods: tuple[str, ...], accent: str) -> str:
        badges = "".join(f"<span>{safe(period)}</span>" for period in periods)
        return (
            f"<div class='snapshot-card {accent}'><strong>{safe(headline)}</strong>"
            f"<p>{safe(sentence)}</p><div class='snapshot-badges'>{badges}</div></div>"
        )

    st.markdown("<div class='snapshot-heading'><span>AT A GLANCE</span><h3>AI Insights Snapshot</h3></div>", unsafe_allow_html=True)
    st.markdown("<div class='snapshot-label'>3 Recurring Strengths</div>", unsafe_allow_html=True)
    for column, item in zip(st.columns(3), snapshot["strengths"]):
        with column:
            st.markdown(snapshot_card(item["headline"], item["sentence"], tuple(item["periods"]), "strength"), unsafe_allow_html=True)

    growth_column, goal_column = st.columns((1.25, 1))
    with growth_column:
        st.markdown("<div class='snapshot-label'>2 Areas of Growth</div>", unsafe_allow_html=True)
        for column, item in zip(st.columns(2), snapshot["growth"]):
            with column:
                st.markdown(snapshot_card(item["headline"], item["sentence"], tuple(item["periods"]), "growth"), unsafe_allow_html=True)
    with goal_column:
        st.markdown("<div class='snapshot-label'>1 Goal &rarr; Growth</div>", unsafe_allow_html=True)
        connection = snapshot["goal_growth"]
        st.markdown(
            f"<div class='snapshot-goal'><div><small>EARLIER GOAL</small><strong>{safe(connection['goal'])}</strong></div>"
            f"<b>&rarr;</b><div><small>LATER EVIDENCE</small><strong>{safe(connection['later_evidence'])}</strong></div>"
            f"<div class='snapshot-badges'><span>{safe(connection['goal_period'])}</span>"
            f"<span>{safe(connection['evidence_period'])}</span></div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='snapshot-label'>3 Teacher Themes</div>", unsafe_allow_html=True)
    for column, theme in zip(st.columns(3), snapshot["teacher_themes"]):
        with column:
            badges = "".join(f"<span>{safe(period)}</span>" for period in theme["periods"])
            st.markdown(
                f"<div class='teacher-theme'><strong>{safe(theme['headline'])}</strong>"
                f"<div class='snapshot-badges'>{badges}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        f"<div class='snapshot-story'><small>MY LEARNING JOURNEY</small>"
        f"<p>{safe(snapshot['story'])}</p></div>",
        unsafe_allow_html=True,
    )
    st.caption("Snapshot evidence comes only from the synthetic Demo Journey.")


def render_visual_insights(cached: dict[str, Any]) -> None:
    parsed = parse_insights(cached["insights"])
    required = {1, 2, 3, 4, 5, 6}
    if not required.issubset(parsed["sections"]):
        st.warning(
            "The cached response is not structured consistently enough for the visual experience. "
            "No content has been reinterpreted or regenerated."
        )
        return

    sections = parsed["sections"]
    overview = st.columns(4)
    overview[0].metric("Recurring strengths", len(sections[1]["items"]))
    overview[1].metric("Growth themes", len(sections[2]["items"]))
    overview[2].metric("Academic themes", len(sections[4]["items"]))
    overview[3].metric("Goal connections", len(sections[6]["items"]))

    st.markdown("### ✨ Strengths That Have Stayed With Me")
    st.caption("Strengths that appeared across more than one reporting period in the cached analysis.")
    render_insight_cards(sections[1]["items"], columns=3, icon="★")

    st.markdown("### 🌱 How I’ve Grown")
    st.caption("Evidence-led growth themes, shown in the order provided by the cached analysis.")
    render_growth_timeline(sections[2]["items"])

    st.markdown("### 🧭 Strengths I’ve Developed")
    st.caption("Learning behaviours and personal strengths that became visible through the journey.")
    render_insight_cards(sections[3]["items"], columns=3, icon="◆")

    st.markdown("### 🎯 Goals → Growth")
    st.caption("Connections shown only where the cached analysis directly linked an earlier goal with later evidence.")
    render_goals_to_growth(sections[6]["items"])

    st.markdown("### 📚 My Academic Story")
    render_insight_cards(sections[4]["items"], columns=2, icon="▰")

    st.markdown("### 💬 What My Teachers Kept Noticing")
    render_insight_cards(sections[5]["items"], columns=3, icon="❝")

    st.markdown("### 🌤️ My Learning Story")
    st.markdown(
        f"<div class='learning-story'>{safe(concise_story(parsed))}</div>",
        unsafe_allow_html=True,
    )
    if sections.get(7, {}).get("items"):
        with st.expander("Important context and limits"):
            for item in sections[7]["items"]:
                st.markdown(f"- {item['summary']}")
                for detail in item.get("evidence", []):
                    st.markdown(f"  - {detail}")


def render_ai_insights(data: dict[str, Any], source_key: str) -> None:
    st.markdown("## AI Learning Insights")
    if source_key == "demo":
        if not DEMO_INSIGHTS_PATH.exists():
            st.error("The synthetic demo insights asset could not be found.")
            return
        demo_insights = load_data(DEMO_INSIGHTS_PATH)
        st.markdown(
            "<div class='demo-ai-label'><strong>Demo Mode</strong> — All learning data and AI insights "
            "shown here are synthetic.</div>",
            unsafe_allow_html=True,
        )
        render_demo_insights_snapshot(demo_insights["snapshot"])
        st.markdown("### Explore the full story")
        st.caption("Open any evidence panel to see the fictional reporting periods behind each theme.")
        render_visual_insights(demo_insights)
        return

    st.write(
        "Generate an evidence-led, strengths-based view across the full learning journey."
    )
    st.markdown(
        "<div class='context-note'><strong>Before you generate:</strong> the exact sanitized and "
        "de-identified learning evidence previewed below will be sent to OpenAI for analysis. "
        "Original PDFs, the reports folder, photographs, names, schools, teachers, and the unsanitized "
        "dataset are never included.</div>",
        unsafe_allow_html=True,
    )
    payload, privacy = prepare_ai_payload(data)
    cache_path = APP_CACHE_DIR / f"openai_insights_{source_key}_{payload_hash(payload)[:12]}.json"

    columns = st.columns(4)
    columns[0].metric("Reporting periods", len(payload["reporting_periods"]))
    columns[1].metric("Characters", f"{privacy['payload_character_count']:,}")
    columns[2].metric("Approx. words", f"{privacy['payload_word_count']:,}")
    columns[3].metric("Privacy status", privacy["status"].replace("_", " ").title())

    st.markdown("### Local privacy scan")
    if privacy["high_risk_match_count"]:
        st.error(
            f"{privacy['high_risk_match_count']} high-risk possible identifier match(es) remain. "
            "The payload must not be used externally."
        )
    elif privacy["review_match_count"]:
        st.warning(
            f"No high-risk identifiers were detected. {privacy['review_match_count']} broad name-like "
            "match(es) need human review because curriculum phrases can trigger false positives."
        )
    else:
        st.success("No identifier patterns were detected by the local scan.")

    removed = privacy.get("removed_counts", {})
    if removed:
        st.caption(
            "Final-stage removals: "
            + " · ".join(f"{label.replace('_', ' ')}: {count}" for label, count in removed.items())
        )
    if privacy["findings"]:
        with st.expander("Privacy flags by category and JSON path"):
            st.dataframe(privacy["findings"], hide_index=True, width="stretch")

    cached = load_cached_insights(cache_path, payload)
    privacy_clear = privacy["status"] == "clear"
    key_ready = api_key_available()
    if not privacy_clear:
        st.error("Generation is blocked because the local privacy scan is not Clear.")
    elif not key_ready:
        st.info(
            "Set OPENAI_API_KEY in the environment before starting the app to enable generation. "
            "The key is never displayed or stored by this application."
        )

    button_label = "Regenerate AI Insights" if cached else "Generate AI Insights"
    generate = st.button(
        button_label,
        type="primary",
        disabled=not (privacy_clear and key_ready),
        key="generate_ai_insights",
    )
    if generate:
        with st.spinner(f"Sending the sanitized evidence in one request to {OPENAI_MODEL}…"):
            try:
                cached = generate_openai_insights(
                    payload,
                    privacy,
                    cache_path,
                )
            except Exception as error:
                st.error(str(error))

    if cached:
        st.caption(
            f"Generated with {safe(cached['model'])} on {safe(cached['generated_at'])}. "
            "This result is cached locally and will not be regenerated automatically."
        )
        render_visual_insights(cached)

    st.markdown("### Preview AI Data")
    st.caption(
        "This is the exact in-memory payload that would be sent after an explicit Generate click. "
        "Review it before enabling any external analysis."
    )
    st.code(payload_json(payload), language="json", line_numbers=True)


def render_landing_page() -> None:
    st.markdown(
        f"<div class='landing-hero'><div class='landing-mark'>{growth_motif('plant', 'growth-motif hero-plant')}</div><div>"
        "<h1>Little Chapters</h1><h2>Every report is a chapter. See the whole learning story.</h2>"
        "<p>Bring reports across the years together to explore academic development, learning behaviours, "
        "teacher observations and meaningful insights.</p></div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='landing-section-title'>Choose how to begin</div>", unsafe_allow_html=True)
    demo_column, upload_column = st.columns(2)
    with demo_column:
        st.markdown(
            "<div class='journey-choice demo-choice'><span>✦</span><h3>Explore a Demo Journey</h3>"
            "<p>Discover the complete experience using entirely synthetic learning data.</p></div>",
            unsafe_allow_html=True,
        )
        if st.button("Explore Demo", type="primary", use_container_width=True, key="enter_demo"):
            st.session_state["journey_mode"] = "demo"
            st.rerun()
    with upload_column:
        st.markdown(
            "<div class='journey-choice upload-choice'><span>↥</span><h3>Create a Journey from Reports</h3>"
            "<p>Upload PDF school reports to build a private, chronological learning journey.</p></div>",
            unsafe_allow_html=True,
        )
        if st.button("Upload Reports", use_container_width=True, key="enter_upload"):
            st.session_state["journey_mode"] = "uploaded"
            st.rerun()

    st.markdown("<div class='landing-section-title'>What you’ll discover</div>", unsafe_allow_html=True)
    features = (
        ("✦", "My Journey", "The story year by year"),
        ("▥", "Academic Journey", "Academic development over time"),
        ("◎", "How I Learn", "Learning habits and behaviours"),
        ("❝", "Through My Teachers’ Eyes", "Teacher observations across the journey"),
        ("◇", "AI Learning Insights", "Themes and connections across years"),
    )
    feature_columns = st.columns(5)
    for column, (icon, title, description) in zip(feature_columns, features):
        with column:
            st.markdown(
                f"<div class='landing-feature'><span>{icon}</span><strong>{safe(title)}</strong>"
                f"<p>{safe(description)}</p></div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        "<div class='privacy-promise'><span>⌾</span><div><strong>Designed with privacy in mind</strong>"
        "<p>Uploaded PDFs are processed in memory and original PDFs are never sent to OpenAI. "
        "Optional AI analysis uses only sanitized structured learning information after a privacy check and explicit action.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )


st.markdown("""
<style>
:root{--ink:#263238;--plum:#65558f;--lavender:#eee9f7;--teal:#318a87;--gold:#f5cf68}
.stApp{background:linear-gradient(180deg,#fffaf1 0,#fff 340px);color:var(--ink)}.block-container{max-width:1180px;padding-top:2rem}h1,h2,h3{letter-spacing:-.025em;color:#2f2b3a}
.hero{padding:1.7rem 2rem;border-radius:24px;background:linear-gradient(120deg,#695894,#8878ad 58%,#4b9691);color:white;margin-bottom:1.2rem;box-shadow:0 14px 40px #5f52731c}.hero h1{color:white;margin:0;font-size:2.45rem}.hero p{margin:.4rem 0 0;opacity:.9;font-size:1.05rem}
.metric-card{background:#fff;border:1px solid #ebe5dc;border-radius:18px;padding:1rem 1.1rem;min-height:128px;box-shadow:0 8px 24px #483f3210;display:flex;flex-direction:column}.metric-card span{color:#716a62;font-size:.82rem;text-transform:uppercase;letter-spacing:.06em}.metric-card strong{color:var(--plum);font-size:2rem;line-height:1.4}.metric-card small{color:#817a72}
.timeline{display:flex;overflow-x:auto;padding:1.5rem .3rem 1rem;margin-bottom:1rem}.timeline-item{position:relative;min-width:145px;padding:1.3rem .7rem .5rem;border-top:3px solid #ded6e9}.timeline-dot{position:absolute;width:14px;height:14px;border-radius:50%;background:#b8abc9;top:-8px;left:.7rem;border:3px solid white;box-shadow:0 0 0 1px #b8abc9}.timeline-item.active{border-color:var(--plum);background:#f8f5fb;border-radius:0 0 14px 14px}.timeline-item.active .timeline-dot{background:var(--gold);box-shadow:0 0 0 2px var(--plum)}.timeline-year{font-weight:700;color:var(--plum)}.timeline-grade{font-weight:650}.timeline-period{color:#777078;font-size:.8rem}
.rating-card,.story-card,.context-note{background:white;border:1px solid #e8e2ef;border-radius:16px;padding:1rem 1.1rem;margin:.7rem 0;box-shadow:0 6px 18px #463e5510}.rating-title{font-weight:700;margin-bottom:.7rem}.period-chip{float:right;background:var(--lavender);color:var(--plum);border-radius:999px;padding:.2rem .65rem;font-size:.75rem;font-weight:600}
.rating-track{display:grid;grid-auto-flow:column;grid-auto-columns:1fr;height:30px;border-radius:8px;overflow:hidden;border:1px solid #d8d0e0}.track-cell{position:relative;border-right:1px solid #ded8e5;background:#faf9fb}.track-cell:last-child{border-right:0}.track-cell.reference{background:#f8dc8b}.current-marker,.previous-marker{position:absolute;width:15px;height:15px;border-radius:50%;top:7px;right:-8px;z-index:3}.current-marker{background:var(--plum);box-shadow:0 0 0 3px white}.previous-marker{background:white;border:3px solid var(--teal);width:13px;height:13px}
.rating-meaning{background:#f8f5fb;border-radius:12px;padding:.65rem .75rem;margin-bottom:.75rem}.rating-meaning strong{color:#4c405d;font-size:.9rem}.rating-meaning p{color:#625b67;font-size:.8rem;line-height:1.4;margin:.18rem 0 .42rem}.rating-badges{display:flex;flex-wrap:wrap;gap:.3rem}.rating-badges span{background:#fff;color:#5e5077;border:1px solid #ddd3e8;border-radius:999px;padding:.16rem .48rem;font-size:.66rem;font-weight:700}.learning-observation{background:linear-gradient(115deg,#f3eef9,#edf8f6);border:1px solid #dcd7e7;border-radius:18px;padding:1rem 1.15rem;margin:.8rem 0 1rem}.learning-observation small{color:#65558f;font-size:.67rem;font-weight:800;letter-spacing:.08em}.learning-observation p{margin:.3rem 0 0;line-height:1.55;color:#40394a}.habit-card{background:#fff;border:1px solid #e4ddec;border-radius:16px;padding:.9rem 1rem;min-height:95px;box-shadow:0 7px 18px #51466710;display:flex;flex-direction:column;gap:.45rem;margin:.25rem 0}.habit-card strong{color:#383142;font-size:1rem}.habit-card span{align-self:flex-start;background:#edf6f5;color:#276f6c;border:1px solid #cee4e2;border-radius:999px;padding:.22rem .58rem;font-size:.73rem;font-weight:700}.story-card{line-height:1.6}.story-meta{color:var(--teal);font-size:.76rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em;margin-bottom:.35rem}.context-note{background:#f2f8f7;border-color:#d5e8e6}.uncertain{color:#766c61;background:#f5f1ea;border-radius:8px;padding:.35rem .55rem;font-size:.78rem}div[data-testid="stMetric"]{background:white;border:1px solid #ebe5dc;border-radius:14px;padding:.7rem 1rem}
.insight-card{background:linear-gradient(145deg,#fff,#faf7fd);border:1px solid #e4ddec;border-radius:18px;padding:1rem;min-height:128px;box-shadow:0 8px 22px #51466712;margin-top:.7rem}.insight-icon{width:34px;height:34px;border-radius:11px;background:#eee8f7;color:var(--plum);display:flex;align-items:center;justify-content:center;font-size:1.05rem;margin-bottom:.7rem}.insight-summary{font-weight:700;line-height:1.4;color:#383142}.evidence-badges{display:flex;flex-wrap:wrap;gap:.3rem;margin:.45rem 0 .2rem}.evidence-badges span{background:#edf6f5;color:#276f6c;border:1px solid #cee4e2;border-radius:999px;padding:.2rem .55rem;font-size:.7rem;font-weight:650}.growth-row{display:flex;gap:1rem;align-items:center;background:white;border:1px solid #e4ddec;border-radius:18px;padding:1rem 1.1rem;margin-top:.8rem}.growth-number{flex:0 0 38px;height:38px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:var(--plum);color:white;font-weight:800}.growth-path{font-size:.75rem;color:#7b7282;margin-top:.3rem}.growth-path span{color:var(--teal);padding:0 .35rem;font-weight:800}.goal-link{display:grid;grid-template-columns:1fr auto 1fr;gap:1rem;align-items:center;background:linear-gradient(90deg,#fff9e8,#fff,#edf7f5);border:1px solid #e6dfd3;border-radius:18px;padding:1rem 1.2rem;margin-top:.8rem}.goal-link>div:not(.goal-arrow){display:flex;flex-direction:column;gap:.25rem}.goal-link small{color:#81766c;font-size:.68rem;font-weight:800;letter-spacing:.06em}.goal-arrow{color:var(--teal);font-size:1.6rem;font-weight:800}.goal-link.single{display:block}.learning-story{background:linear-gradient(120deg,#695894,#4b9691);color:white;border-radius:22px;padding:1.4rem 1.6rem;font-size:1.08rem;line-height:1.7;box-shadow:0 12px 30px #52456920}
.source-banner{border-radius:16px;padding:.85rem 1rem;margin:.7rem 0 1rem;border:1px solid}.source-banner.demo{background:#f5effb;border-color:#dfd2ec;color:#57436f}.source-banner.uploaded{background:#edf8f6;border-color:#cce5e1;color:#286864}
.landing-hero{display:grid;grid-template-columns:auto 1fr;gap:1.2rem;align-items:start;padding:2.4rem 2.5rem;border-radius:28px;background:linear-gradient(125deg,#67558f 0,#7c68a5 52%,#438f8c 100%);color:#fff;box-shadow:0 18px 45px #51436d24;margin-bottom:1.5rem}.landing-mark{width:58px;height:58px;border-radius:18px;background:#ffffff20;display:flex;align-items:center;justify-content:center;color:#f7d273;font-size:2rem}.landing-hero h1{font-size:2.8rem;color:#fff;margin:-.2rem 0 0}.landing-hero h2{font-size:1.45rem;color:#fff;margin:.15rem 0 .55rem;font-weight:650}.landing-hero p{max-width:760px;margin:0;line-height:1.55;opacity:.9}.landing-section-title{font-size:1.05rem;font-weight:800;color:#3b3541;margin:1.25rem 0 .5rem}.journey-choice{border-radius:21px;padding:1.25rem 1.35rem;min-height:170px;border:1px solid;box-shadow:0 9px 25px #51466712}.journey-choice.demo-choice{background:linear-gradient(145deg,#f7f2fc,#fff);border-color:#ded3eb}.journey-choice.upload-choice{background:linear-gradient(145deg,#edf8f6,#fff);border-color:#cee5e1}.journey-choice>span{width:38px;height:38px;border-radius:12px;background:#fff;display:flex;align-items:center;justify-content:center;color:#65558f;font-size:1.2rem}.journey-choice h3{margin:.65rem 0 .3rem}.journey-choice p{margin:0;color:#6b646d;line-height:1.45}.landing-feature{background:#fff;border:1px solid #e5dfea;border-radius:16px;padding:.85rem;min-height:132px;box-shadow:0 6px 17px #5146670c}.landing-feature>span{display:block;color:#65558f;font-size:1.1rem;margin-bottom:.45rem}.landing-feature strong{font-size:.82rem;line-height:1.25}.landing-feature p{font-size:.7rem;line-height:1.35;color:#756e78;margin:.3rem 0 0}.privacy-promise{display:flex;gap:.8rem;align-items:flex-start;background:#f1f8f7;border:1px solid #d2e7e4;border-radius:18px;padding:1rem 1.2rem;margin-top:1.25rem;color:#356563}.privacy-promise>span{font-size:1.3rem}.privacy-promise p{margin:.2rem 0 0;font-size:.8rem;line-height:1.45;color:#55736f}.selected-period{margin:1.2rem 0 .8rem}.selected-period small{color:#65558f;font-size:.68rem;font-weight:800;letter-spacing:.09em}.selected-period h2{margin:.15rem 0 0}.journey-band-title{font-size:1.05rem;font-weight:800;color:#3b3541;background:#fff8e8;border:1px solid #f3dfb9;border-bottom:0;border-radius:18px 18px 0 0;padding:.8rem 1rem;margin-top:.35rem}.highlight-card{display:flex;gap:.7rem;align-items:center;background:#fffdf9;border:1px solid #f0dfc4;border-radius:15px;padding:.8rem .9rem;min-height:92px;box-shadow:0 6px 16px #6c57300b}.highlight-card span{flex:0 0 34px;height:34px;border-radius:50%;background:#f5eafa;color:#7052a0;display:flex;align-items:center;justify-content:center;font-size:1.15rem}.highlight-card p{margin:0;font-size:.82rem;line-height:1.4}.quote-card,.approach-card{border-radius:17px;padding:1rem 1.1rem;margin:.45rem 0;min-height:125px}.quote-card{display:flex;gap:.8rem;background:#fff;border:1px solid #dce5f1;box-shadow:0 7px 20px #45556d0d}.quote-card span{color:#4387d8;font-size:2rem;font-weight:800;line-height:1}.quote-card p,.approach-card p{margin:0;line-height:1.55}.approach-card{background:linear-gradient(140deg,#f3eefb,#edf8f6);border:1px solid #dcd7e8}.approach-card small{color:#65558f;font-size:.64rem;font-weight:800;letter-spacing:.08em}.approach-card p{margin-top:.35rem}.journey-indicator{background:#fff;border:1px solid #e3ddec;border-radius:15px;padding:.85rem .95rem;min-height:92px;box-shadow:0 6px 16px #5146670d;display:flex;flex-direction:column;gap:.45rem}.journey-indicator span{align-self:flex-start;background:#edf6f5;color:#276f6c;border:1px solid #cee4e2;border-radius:999px;padding:.18rem .52rem;font-size:.7rem;font-weight:700}.learning-observation{display:flex;gap:.8rem;align-items:flex-start}.learning-observation-icon,.habit-icon{flex:0 0 34px;height:34px;border-radius:11px;background:#fff;color:#318a87;display:flex;align-items:center;justify-content:center}.habit-icon{background:#f1ebf8;color:#65558f;margin-bottom:.15rem}.teacher-journey-line{height:3px;background:linear-gradient(90deg,#65558f,#9b86cf,#4b9691);border-radius:999px;margin:1rem 0 1.3rem}.teacher-period-heading{display:flex;gap:.8rem;align-items:center;margin:1.3rem 0 .45rem}.teacher-period-dot{width:40px;height:40px;border-radius:50%;background:#eee8f7;color:#65558f;display:flex;align-items:center;justify-content:center;box-shadow:0 0 0 5px #faf8fd}.teacher-period-heading span{color:#65558f;font-size:.72rem;font-weight:800}.teacher-period-heading h3{margin:-.05rem 0}.teacher-period-heading small{color:#756d79}.teacher-card-label{display:inline-block;border-radius:999px;padding:.25rem .55rem;font-size:.65rem;font-weight:800;letter-spacing:.05em;margin-bottom:.3rem}.teacher-card-label.observation{background:#eaf3fd;color:#2f72bd}.teacher-card-label.goal{background:#fff3d9;color:#8c641a}.teacher-quote-card,.teacher-goal-card{border-radius:16px;padding:.9rem 1rem;min-height:115px;margin-bottom:.55rem}.teacher-quote-card{display:flex;gap:.75rem;background:#fff;border:1px solid #dce5f1;box-shadow:0 6px 18px #45556d0d}.teacher-quote-card span{color:#4387d8;font-size:1.8rem;font-weight:800;line-height:1}.teacher-quote-card p,.teacher-goal-card p{margin:0;line-height:1.5}.teacher-goal-card{background:linear-gradient(145deg,#fffaf0,#fff);border:1px solid #f0dfbd}.academic-period{display:grid;grid-template-columns:70px 1fr auto;align-items:center;gap:.7rem;border-left:4px solid #8b72c2;padding:.55rem .8rem;margin:1.1rem 0 .2rem;background:#faf8fc;border-radius:0 12px 12px 0}.academic-period span{color:#65558f;font-weight:800}.academic-period small{color:#756d79}.academic-summary-card{background:linear-gradient(145deg,#fff,#faf8fd);border:1px solid #e1dbe9;border-radius:17px;padding:.9rem 1rem;min-height:190px;box-shadow:0 7px 20px #5146670e}.academic-summary-card>small{color:#318a87;font-size:.65rem;font-weight:800;letter-spacing:.06em;text-transform:uppercase}.academic-summary-card h4{margin:.25rem 0 .55rem;color:#332d3b}.academic-summary-card>strong{font-size:.82rem;color:#584969}.academic-summary-card p{font-size:.76rem;line-height:1.4;color:#6b636d;margin:.3rem 0 .55rem}.demo-ai-label{display:inline-block;background:#f3eef9;color:#59466f;border:1px solid #dfd3ec;border-radius:999px;padding:.38rem .72rem;font-size:.76rem;margin:.15rem 0 .35rem}.snapshot-heading{display:flex;align-items:baseline;gap:.65rem;margin:1rem 0 .25rem}.snapshot-heading span,.snapshot-label{color:#76698c;font-size:.67rem;font-weight:800;letter-spacing:.09em;text-transform:uppercase}.snapshot-heading h3{margin:0}.snapshot-label{margin:.55rem 0 .15rem}.snapshot-card{border-radius:15px;padding:.72rem .8rem;min-height:118px;border:1px solid #e3dbea;box-shadow:0 5px 15px #5146670d}.snapshot-card.strength{background:linear-gradient(145deg,#fff,#f8f3fc)}.snapshot-card.growth{background:linear-gradient(145deg,#fff,#eef8f6);border-color:#d6e8e5}.snapshot-card strong,.teacher-theme strong{display:block;color:#393143;font-size:.9rem}.snapshot-card p{font-size:.76rem;line-height:1.35;color:#665f69;margin:.3rem 0 .5rem}.snapshot-badges{display:flex;flex-wrap:wrap;gap:.22rem;margin-top:.42rem}.snapshot-badges span{background:#fff;border:1px solid #ded6e5;color:#65558f;border-radius:999px;padding:.12rem .38rem;font-size:.58rem;font-weight:700}.snapshot-goal{display:grid;grid-template-columns:1fr auto 1fr;gap:.55rem;align-items:center;background:linear-gradient(90deg,#fff8e7,#eef8f6);border:1px solid #e5dfd2;border-radius:15px;padding:.72rem .8rem;min-height:118px}.snapshot-goal>div:not(.snapshot-badges){display:flex;flex-direction:column}.snapshot-goal small,.snapshot-story small{color:#81736a;font-size:.57rem;font-weight:800;letter-spacing:.07em}.snapshot-goal strong{font-size:.74rem;line-height:1.25;color:#3d3741}.snapshot-goal>b{color:var(--teal);font-size:1.2rem}.snapshot-goal .snapshot-badges{grid-column:1/-1;margin-top:0}.teacher-theme{background:#fff;border:1px solid #e2ddeb;border-radius:13px;padding:.6rem .72rem;min-height:70px;box-shadow:0 4px 12px #5146670b}.snapshot-story{margin-top:.65rem;background:linear-gradient(110deg,#65558f,#4b9691);color:#fff;border-radius:16px;padding:.75rem 1rem}.snapshot-story small{color:#eee9f7}.snapshot-story p{margin:.2rem 0 0;font-size:.84rem;line-height:1.4}
.timeline-growth{height:28px;margin-bottom:.2rem;color:#7f6db0}.growth-motif.small{width:27px;height:27px}.timeline-item.active .timeline-growth{color:#318a87}.landing-hero{grid-template-columns:auto 1fr;gap:1.45rem;align-items:center}.landing-mark{width:88px;height:88px;border-radius:24px;background:#ffffff18;color:#bff1d9;box-shadow:inset 0 0 0 1px #ffffff24}.growth-motif.hero-plant{width:72px;height:72px;color:#72d3a4;filter:drop-shadow(0 6px 10px #26323825)}.hero-plant .plant-stem,.hero-plant .plant-root{stroke:#d5f5df}.hero-plant .plant-leaf{stroke:#daf7e4;stroke-width:2.4}.hero-plant .leaf-left{fill:#42c98b}.hero-plant .leaf-right{fill:#8be2a7}
@media(max-width:700px){.hero h1{font-size:1.9rem}.metric-card{min-height:105px}.period-chip{float:none;display:inline-block;margin-left:.3rem}.snapshot-goal{grid-template-columns:1fr}.snapshot-goal>b{transform:rotate(90deg);text-align:center}.snapshot-goal .snapshot-badges{grid-column:1}.landing-hero{grid-template-columns:1fr;padding:1.6rem}.landing-mark{width:64px;height:64px}.growth-motif.hero-plant{width:52px;height:52px}}
</style>""", unsafe_allow_html=True)

journey_mode = st.session_state.get("journey_mode")
if journey_mode not in {"demo", "uploaded"}:
    render_landing_page()
    st.stop()

top_column, change_column = st.columns((5, 1))
with top_column:
    st.markdown("<div class='hero'><h1>Little Chapters</h1><p>Every report is a chapter. See the whole learning story.</p></div>", unsafe_allow_html=True)
with change_column:
    if st.button("← Change journey", use_container_width=True, key="change_journey"):
        st.session_state.pop("journey_mode", None)
        st.rerun()

if journey_mode == "demo":
    if not DEMO_DATA_PATH.exists():
        st.error("The synthetic demo dataset could not be found.")
        st.stop()
    data = load_data(DEMO_DATA_PATH)
    source_key = "demo"
    st.markdown(
        "<div class='source-banner demo'><strong>Demo Mode</strong> — This application uses entirely "
        "synthetic learning data. No real student information is included.</div>",
        unsafe_allow_html=True,
    )
else:
    data, _upload_summaries = render_upload_workflow()
    source_key = "uploaded"
    if data is None:
        st.stop()

reports = data.get("reports", [])
if not reports:
    st.warning("The local dataset contains no reports to display.")
    st.stop()

render_overview(reports)
st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
views = ("My Journey", "Academic Journey", "How I Learn", "Through My Teachers’ Eyes", "AI Learning Insights")
view = st.radio("Explore", views, horizontal=True, key="main_view")
if view == "My Journey":
    if source_key == "demo":
        render_demo_my_journey(reports)
    else:
        render_my_journey(reports)
elif view == "Academic Journey":
    if source_key == "demo":
        render_demo_academic_journey(reports)
    else:
        render_academic_journey(reports)
elif view == "How I Learn":
    render_how_i_learn(reports, demo_mode=source_key == "demo")
elif view == "Through My Teachers’ Eyes":
    if source_key == "demo":
        render_demo_teachers_eyes(reports)
    else:
        render_teachers_eyes(reports)
else:
    render_ai_insights(data, source_key)
st.markdown("---")
st.caption("Deterministic extraction runs locally · Optional OpenAI analysis only after privacy review and explicit consent")
