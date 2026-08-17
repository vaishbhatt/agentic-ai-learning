"""Local prototype for graphical assessment-rating extraction.

Scope is deliberately fixed to the two approved samples:
  * first PDF in reports/ (sorted by filename), page 3
  * second PDF in reports/ (sorted by filename), page 2

The detector uses PDF vector geometry first and rendered pixels only to validate
the selected cell. Output never includes extracted text or full report pages.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import pymupdf
from PIL import Image, ImageDraw


REPORTS_DIR = Path("reports")
OUTPUT_DIR = Path("rating_prototype_output")
RENDER_SCALE = 2.0  # 144 DPI for normal 72-DPI PDF coordinates.
SAMPLES = ((1, 3, "format_1"), (2, 2, "format_2"))


@dataclass(frozen=True)
class Cell:
    rect: pymupdf.Rect
    fill: tuple[float, float, float]


def color_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def cluster_by_y(cells: list[Cell], tolerance: float = 1.5) -> list[list[Cell]]:
    rows: list[list[Cell]] = []
    for cell in sorted(cells, key=lambda item: (item.rect.y0, item.rect.x0)):
        matching = next(
            (row for row in rows if abs(row[0].rect.y0 - cell.rect.y0) <= tolerance),
            None,
        )
        if matching is None:
            rows.append([cell])
        else:
            matching.append(cell)
    return [sorted(row, key=lambda item: item.rect.x0) for row in rows]


def candidate_rows(page: pymupdf.Page) -> list[list[Cell]]:
    cells: list[Cell] = []
    for drawing in page.get_drawings():
        fill = drawing.get("fill")
        if fill is None:
            continue
        rect = pymupdf.Rect(drawing["rect"])
        if (
            rect.x0 > page.rect.width * 0.25
            and 20 <= rect.width <= 50
            and 12 <= rect.height <= 55
        ):
            cells.append(Cell(rect=rect, fill=tuple(float(value) for value in fill[:3])))

    rows = []
    for row in cluster_by_y(cells):
        widths = [cell.rect.width for cell in row]
        heights = [cell.rect.height for cell in row]
        if not 8 <= len(row) <= 12:
            continue
        if max(widths) - min(widths) > 1.0 or max(heights) - min(heights) > 1.0:
            continue
        gaps = [row[index + 1].rect.x0 - row[index].rect.x0 for index in range(len(row) - 1)]
        if gaps and max(gaps) - min(gaps) <= 1.0:
            rows.append(row)
    return rows


def render_row(page: pymupdf.Page, row: list[Cell]) -> tuple[Image.Image, list[tuple[int, int, int, int]]]:
    clip = pymupdf.Rect(row[0].rect.x0, row[0].rect.y0, row[-1].rect.x1, row[0].rect.y1)
    pixmap = page.get_pixmap(
        matrix=pymupdf.Matrix(RENDER_SCALE, RENDER_SCALE),
        clip=clip,
        colorspace=pymupdf.csRGB,
        alpha=False,
    )
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    boxes = []
    for cell in row:
        boxes.append(
            (
                round((cell.rect.x0 - clip.x0) * RENDER_SCALE),
                round((cell.rect.y0 - clip.y0) * RENDER_SCALE),
                round((cell.rect.x1 - clip.x0) * RENDER_SCALE),
                round((cell.rect.y1 - clip.y0) * RENDER_SCALE),
            )
        )
    return image, boxes


def dark_pixel_count(image: Image.Image, center_x: int, half_width: int) -> int:
    left = max(0, center_x - half_width)
    right = min(image.width, center_x + half_width + 1)
    top = max(0, round(image.height * 0.12))
    bottom = min(image.height, round(image.height * 0.88))
    grey = image.crop((left, top, right, bottom)).convert("L")
    histogram = grey.histogram()
    return sum(histogram[:80])


def analyse_row(page: pymupdf.Page, row: list[Cell], row_number: int) -> tuple[dict[str, Any], Image.Image]:
    rendered, boxes = render_row(page, row)
    # Rating positions are encoded at the right boundary of each scale cell.
    # Count very dark rendered pixels around each boundary. A filled black dot
    # produces substantially more dark area than grid rules, dashed lines, or
    # the hollow starting marker used by the newer format.
    half_width = max(5, round((boxes[0][2] - boxes[0][0]) * 0.22))
    marker_scores = [dark_pixel_count(rendered, box[2], half_width) for box in boxes]
    selected_index = max(range(len(row)), key=marker_scores.__getitem__)
    ordered_scores = sorted(marker_scores, reverse=True)
    marker_margin = ordered_scores[0] - ordered_scores[1]
    baseline = median(marker_scores)
    marker_excess = marker_scores[selected_index] - baseline

    # Vector fill identifies the yellow reference/progression band, not the
    # student's black-dot marker. Preserve it only as useful structural data.
    fill_scores = [color_distance(cell.fill, (1.0, 1.0, 1.0)) for cell in row]
    reference_index = max(range(len(row)), key=fill_scores.__getitem__)
    completeness = min(1.0, len(row) / 10.0)
    marker_strength = min(1.0, max(0.0, marker_margin) / 100.0)
    excess_strength = min(1.0, max(0.0, marker_excess) / 150.0)
    confidence = round(0.40 * completeness + 0.35 * marker_strength + 0.25 * excess_strength, 3)

    draw = ImageDraw.Draw(rendered)
    for index, box in enumerate(boxes):
        draw.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), outline=(40, 120, 220), width=1)
    marker_x = boxes[selected_index][2]
    draw.rectangle(
        (marker_x - half_width, 1, marker_x + half_width, rendered.height - 2),
        outline=(0, 170, 0),
        width=3,
    )

    result = {
        "row": row_number,
        "selected_position": selected_index + 1,
        "number_of_positions": len(row),
        "confidence": confidence,
        "marker_source": "rendered_filled_circle_with_pdf_cell_geometry",
        "reference_band_position": reference_index + 1,
        "rendered_pixel_validation": {
            "filled_circle_detected": marker_excess > 50 and marker_margin > 30,
            "dark_pixel_score": marker_scores[selected_index],
            "next_best_margin": marker_margin,
        },
        "scale_bbox_points": [round(value, 2) for value in (
            row[0].rect.x0,
            row[0].rect.y0,
            row[-1].rect.x1,
            row[0].rect.y1,
        )],
    }
    return result, rendered


def make_debug_composite(strips: list[Image.Image], selected_positions: list[int]) -> Image.Image:
    label_width = 170
    gap = 12
    width = label_width + max(strip.width for strip in strips)
    height = sum(strip.height for strip in strips) + gap * (len(strips) + 1)
    output = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(output)
    y = gap
    for index, (strip, selected) in enumerate(zip(strips, selected_positions), 1):
        draw.text((8, y + max(0, strip.height // 2 - 6)), f"row {index}: position {selected}", fill="black")
        output.paste(strip, (label_width, y))
        y += strip.height + gap
    return output


def process_sample(pdf: Path, document_number: int, page_number: int, format_name: str) -> dict[str, Any]:
    with pymupdf.open(pdf) as document:
        if page_number > document.page_count:
            raise ValueError(f"approved page {page_number} does not exist in document {document_number}")
        page = document[page_number - 1]
        rows = candidate_rows(page)
        if not rows:
            raise RuntimeError(f"no assessment rows detected in document {document_number}")

        row_results: list[dict[str, Any]] = []
        strips: list[Image.Image] = []
        for row_number, row in enumerate(rows, 1):
            result, strip = analyse_row(page, row, row_number)
            row_results.append(result)
            strips.append(strip)

        debug_name = f"{format_name}_page_{page_number}_debug.png"
        debug = make_debug_composite(
            strips,
            [result["selected_position"] for result in row_results],
        )
        debug.save(OUTPUT_DIR / debug_name)

        return {
            "document": document_number,
            "page": page_number,
            "format": format_name,
            "detected_rows": len(row_results),
            "debug_image": debug_name,
            "rows": row_results,
        }


def main() -> int:
    pdfs = sorted(REPORTS_DIR.glob("*.pdf"))
    if len(pdfs) < 2:
        raise SystemExit("Expected at least two PDF files in reports/")

    OUTPUT_DIR.mkdir(exist_ok=True)
    output = {
        "method": "PDF vector cell geometry with 144-DPI rendered black-dot detection",
        "privacy": "No text extraction, OCR, full-page images, or external processing.",
        "samples": [],
    }
    for document_number, page_number, format_name in SAMPLES:
        output["samples"].append(
            process_sample(pdfs[document_number - 1], document_number, page_number, format_name)
        )

    json_path = OUTPUT_DIR / "detected_ratings.json"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}")
    for sample in output["samples"]:
        agreements = sum(
            row["rendered_pixel_validation"]["filled_circle_detected"]
            for row in sample["rows"]
        )
        print(
            f"{sample['format']}: page={sample['page']} rows={sample['detected_rows']} "
            f"filled_circle_detections={agreements}/{sample['detected_rows']} "
            f"debug={sample['debug_image']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
