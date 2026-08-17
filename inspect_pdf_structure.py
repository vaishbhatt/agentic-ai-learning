"""Print structural PDF diagnostics without printing document text.

This tool runs locally and only reads PDF files from the requested directory.
It never writes report contents or extracts embedded images.
"""

from __future__ import annotations

import argparse
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pymupdf


CHECKMARK_CODEPOINTS = {
    0x2611,  # ballot box with check
    0x2705,  # white heavy check mark
    0x2713,  # check mark
    0x2714,  # heavy check mark
    0x2717,  # ballot X
    0x2718,  # heavy ballot X
}


def fmt_rect(value: Any) -> str:
    rect = pymupdf.Rect(value)
    return f"({rect.x0:.2f}, {rect.y0:.2f}, {rect.x1:.2f}, {rect.y1:.2f})"


def item_bbox(item: tuple[Any, ...]) -> pymupdf.Rect | None:
    """Return an approximate bounding box for one get_drawings() path item."""
    points: list[pymupdf.Point] = []
    for value in item[1:]:
        if isinstance(value, pymupdf.Point):
            points.append(value)
        elif isinstance(value, (pymupdf.Rect, pymupdf.Quad)):
            return pymupdf.Rect(value)
    if not points:
        return None
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    return pymupdf.Rect(min(xs), min(ys), max(xs), max(ys))


def symbol_diagnostics(page: pymupdf.Page) -> list[tuple[str, pymupdf.Rect]]:
    """Locate checkmark-like and other Unicode symbol glyphs without returning text."""
    found: list[tuple[str, pymupdf.Rect]] = []
    raw = page.get_text("rawdict")
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                for char in span.get("chars", []):
                    value = char.get("c", "")
                    if not value:
                        continue
                    codepoint = ord(value[0])
                    if codepoint in CHECKMARK_CODEPOINTS:
                        kind = "checkmark-like glyph"
                    elif unicodedata.category(value[0]).startswith("S"):
                        kind = "Unicode symbol glyph"
                    else:
                        continue
                    found.append((kind, pymupdf.Rect(char["bbox"])))
    return found


def describe_page(page: pymupdf.Page) -> None:
    print(f"  Page {page.number + 1}: size={page.rect.width:.2f}x{page.rect.height:.2f}")

    blocks = page.get_text("blocks")
    text_blocks = [block for block in blocks if len(block) > 6 and block[6] == 0]
    image_blocks = [block for block in blocks if len(block) > 6 and block[6] == 1]
    print(f"    text_blocks: {len(text_blocks)}")
    for index, block in enumerate(text_blocks, 1):
        # block[4] contains text; deliberately neither print nor retain it.
        print(f"      [{index}] bbox={fmt_rect(block[:4])}")

    print(f"    image_blocks: {len(image_blocks)}")
    for index, block in enumerate(image_blocks, 1):
        print(f"      [{index}] bbox={fmt_rect(block[:4])}")

    images = page.get_images(full=True)
    print(f"    embedded_images: {len(images)}")
    for index, image in enumerate(images, 1):
        xref = image[0]
        rects = page.get_image_rects(xref)
        locations = ", ".join(fmt_rect(rect) for rect in rects) or "not displayed on page"
        print(
            f"      [{index}] xref={xref} pixel_size={image[2]}x{image[3]} "
            f"bits_per_component={image[4]} placements={len(rects)} bboxes={locations}"
        )

    try:
        tables = page.find_tables()
        print(f"    tables: {len(tables.tables)}")
        for index, table in enumerate(tables.tables, 1):
            print(
                f"      [{index}] bbox={fmt_rect(table.bbox)} "
                f"rows={table.row_count} columns={table.col_count}"
            )
    except Exception as error:  # Keep diagnostics running on unusual PDFs.
        print(f"    tables: detection_failed ({type(error).__name__})")

    drawings = page.get_drawings()
    print(f"    vector_drawings: {len(drawings)}")
    drawing_types = Counter(str(drawing.get("type", "unknown")) for drawing in drawings)
    print(f"      path_types={dict(sorted(drawing_types.items()))}")

    lines: list[tuple[pymupdf.Rect, str]] = []
    filled_areas: list[pymupdf.Rect] = []
    circles_or_dots: list[tuple[pymupdf.Rect, str]] = []

    for drawing_index, drawing in enumerate(drawings, 1):
        drawing_rect = pymupdf.Rect(drawing["rect"])
        dash_value = str(drawing.get("dashes") or "").strip()
        stroke_style = "dashed/dotted" if dash_value and dash_value not in {"[] 0", "[]0"} else "solid"
        item_kinds = Counter(str(item[0]) for item in drawing.get("items", []))
        print(
            f"      [{drawing_index}] bbox={fmt_rect(drawing_rect)} "
            f"type={drawing.get('type', 'unknown')} items={dict(sorted(item_kinds.items()))} "
            f"stroke_style={stroke_style} stroke_width={drawing.get('width')} "
            f"has_fill={drawing.get('fill') is not None}"
        )

        if drawing.get("fill") is not None and drawing_rect.get_area() > 0:
            filled_areas.append(drawing_rect)

        for item in drawing.get("items", []):
            kind = item[0]
            bbox = item_bbox(item)
            if bbox is None:
                continue
            if kind == "l":
                lines.append((bbox, stroke_style))
            elif kind == "re" and drawing.get("fill") is not None:
                filled_areas.append(bbox)

        has_curves = any(item[0] in {"c", "qu"} for item in drawing.get("items", []))
        width, height = abs(drawing_rect.width), abs(drawing_rect.height)
        nearly_square = max(width, height) > 0 and abs(width - height) / max(width, height) <= 0.20
        if has_curves and nearly_square:
            label = "dot candidate" if max(width, height) <= 8 else "circle/ellipse candidate"
            circles_or_dots.append((drawing_rect, label))

    print(f"    lines: {len(lines)}")
    for index, (bbox, style) in enumerate(lines, 1):
        print(f"      [{index}] bbox={fmt_rect(bbox)} style={style}")

    # Deduplicate rectangles reported once as a whole path and once as an item.
    unique_fills = {tuple(round(value, 3) for value in rect): rect for rect in filled_areas}
    print(f"    filled_rectangles_or_areas: {len(unique_fills)}")
    for index, rect in enumerate(unique_fills.values(), 1):
        print(f"      [{index}] bbox={fmt_rect(rect)} area={rect.get_area():.2f}")

    print(f"    circles_or_dots: {len(circles_or_dots)}")
    for index, (rect, label) in enumerate(circles_or_dots, 1):
        print(f"      [{index}] type={label} bbox={fmt_rect(rect)}")

    symbols = symbol_diagnostics(page)
    print(f"    checkmarks_or_symbols: {len(symbols)}")
    for index, (kind, rect) in enumerate(symbols, 1):
        print(f"      [{index}] type={kind} bbox={fmt_rect(rect)}")


def pdf_files(directory: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(path for path in directory.glob(pattern) if path.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect local PDF structure without printing report text."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path("reports"),
        help="directory containing PDFs (default: reports)",
    )
    parser.add_argument("--recursive", action="store_true", help="include subdirectories")
    args = parser.parse_args()

    directory = args.directory.resolve()
    if not directory.is_dir():
        parser.error(f"directory does not exist: {directory}")

    files = list(pdf_files(directory, args.recursive))
    print(f"Directory: {directory}")
    print(f"PDF files found: {len(files)}")
    print("Text content and image data are intentionally omitted.")

    for file_index, path in enumerate(files, 1):
        print(f"\nPDF {file_index}: {path.name}")
        try:
            with pymupdf.open(path) as document:
                print(
                    f"  pages={document.page_count} encrypted={document.is_encrypted} "
                    f"pdf={document.is_pdf}"
                )
                for page in document:
                    describe_page(page)
        except Exception as error:
            print(f"  inspection_failed: {type(error).__name__}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
