
import json
import re

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.enum.shapes import PP_PLACEHOLDER

from config import INPUT_DIR, RECORDS_FILE


# --------------------------------------------------
# 1. TEXT UTILITIES
# --------------------------------------------------

def clean_text(text):
    """Normalize whitespace in extracted text."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def iter_shapes(shapes):
    """Yield shapes recursively, including grouped shapes."""
    for shape in shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from iter_shapes(shape.shapes)
        else:
            yield shape


def get_shape_text(shape):
    """Extract editable text from a PowerPoint shape."""
    if not getattr(shape, "has_text_frame", False):
        return ""

    return clean_text(shape.text)


def get_shape_font_size(shape):
    """Return the largest explicit font size in a text shape."""
    sizes = []

    if not getattr(shape, "has_text_frame", False):
        return 0.0

    for paragraph in shape.text_frame.paragraphs:
        if paragraph.font.size is not None:
            sizes.append(paragraph.font.size.pt)

        for run in paragraph.runs:
            if run.font.size is not None:
                sizes.append(run.font.size.pt)

    return max(sizes, default=0.0)


def get_placeholder_type(shape):
    """Safely return a shape's placeholder type."""
    if not getattr(shape, "is_placeholder", False):
        return None

    try:
        return shape.placeholder_format.type
    except (ValueError, AttributeError):
        return None


# --------------------------------------------------
# 2. SLIDE TITLE EXTRACTION
# --------------------------------------------------

def get_slide_title_details(slide, slide_height):
    """
    Return (title, method).

    Priority:
    1. Dedicated PowerPoint title placeholder.
    2. Subtitle placeholder on title slides.
    3. Heuristic based on font size, position and text length.
    4. First non-empty text as a last resort.

    Note: Heuristics cannot guarantee the true title when a
    slide has no dedicated title placeholder.
    """

    shapes = list(iter_shapes(slide.shapes))

    # Priority 1: actual title placeholder.
    for shape in shapes:
        placeholder_type = get_placeholder_type(shape)

        if placeholder_type in (
            PP_PLACEHOLDER.TITLE,
            PP_PLACEHOLDER.CENTER_TITLE,
        ):
            title = get_shape_text(shape)

            if title:
                return title, "title_placeholder"

    # Priority 2: subtitle placeholder.
    # Useful for title slides, but less reliable on content slides.
    for shape in shapes:
        placeholder_type = get_placeholder_type(shape)

        if placeholder_type == PP_PLACEHOLDER.SUBTITLE:
            title = get_shape_text(shape)

            if title:
                return title, "subtitle_placeholder"

    # Priority 3: rank text boxes as possible headings.
    candidates = []
    slide_height = max(int(slide_height), 1)

    for shape in shapes:
        text = get_shape_text(shape)

        if not text:
            continue

        # Avoid long body paragraphs and large text blocks.
        if len(text) > 180:
            continue

        font_size = get_shape_font_size(shape)

        # Ignore tiny text, such as footers and page numbers.
        if font_size and font_size < 9:
            continue

        top_fraction = max(0, int(shape.top)) / slide_height

        # Short text is more likely to be a heading.
        if len(text) <= 60:
            length_score = 5
        elif len(text) <= 100:
            length_score = 2
        else:
            length_score = 0

        # Prefer text closer to the top.
        position_score = max(0, 10 * (1 - top_fraction))

        # Prefer larger text, but don't let size decide everything.
        font_score = min(font_size, 40) / 2 if font_size else 0

        # Sentences often represent body text.
        punctuation_penalty = (
            2 if text.endswith((".", ",", ";")) else 0
        )

        # Small preference for heading-like text.
        score = (
            font_score
            + position_score
            + length_score
            - punctuation_penalty
        )

        candidates.append({
            "text": text,
            "score": score,
            "top": int(shape.top),
            "font_size": font_size,
        })

    if candidates:
        candidates.sort(
            key=lambda item: (
                item["score"],
                -item["top"],
                -len(item["text"]),
            ),
            reverse=True,
        )

        return candidates[0]["text"], "text_shape_heuristic"

    # Priority 4: last resort.
    for shape in shapes:
        text = get_shape_text(shape)

        if text:
            return text[:150], "first_text_fallback"

    return "Untitled slide", "no_text_found"


def get_slide_title(slide, slide_height):
    """Return only the selected slide title."""
    title, _ = get_slide_title_details(slide, slide_height)
    return title


# --------------------------------------------------
# 3. PRESENTATION TITLE
# --------------------------------------------------

def get_presentation_title(presentation):
    """Infer a presentation title from its first slide."""

    if not presentation.slides:
        return "Untitled presentation"

    first_slide = presentation.slides[0]

    # Prefer the first slide's actual title placeholder.
    for shape in iter_shapes(first_slide.shapes):
        placeholder_type = get_placeholder_type(shape)

        if placeholder_type in (
            PP_PLACEHOLDER.TITLE,
            PP_PLACEHOLDER.CENTER_TITLE,
        ):
            title = get_shape_text(shape)

            if title:
                return title

    # Fallback: use the first reasonably short text.
    for shape in iter_shapes(first_slide.shapes):
        text = get_shape_text(shape)

        if text and len(text) <= 180:
            return text

    return "Untitled presentation"


# --------------------------------------------------
# 4. SLIDE CONTENT EXTRACTION
# --------------------------------------------------

def extract_slide(
    slide,
    source_file,
    slide_number,
    deck_title,
    slide_height,
):
    """Extract one slide into a structured dictionary."""

    title, title_method = get_slide_title_details(
        slide, slide_height
    )

    text_elements = []
    tables = []
    images = []

    for shape in iter_shapes(slide.shapes):

        # Extract editable text.
        text = get_shape_text(shape)

        if text:
            text_elements.append(text)

        # Extract table cells.
        if getattr(shape, "has_table", False):
            table_data = []

            for row in shape.table.rows:
                table_data.append([
                    clean_text(cell.text)
                    for cell in row.cells
                ])

            tables.append(table_data)

        # Record embedded pictures.
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            images.append({
                "shape_name": shape.name,
                "width": shape.width,
                "height": shape.height,
            })

    # Extract speaker notes when available.
    notes = ""

    if slide.has_notes_slide:
        try:
            notes = clean_text(
                slide.notes_slide.notes_text_frame.text
            )
        except (AttributeError, ValueError):
            notes = ""

    # Flatten table content into searchable text.
    table_text = []

    for table in tables:
        for row in table:
            for cell in row:
                if cell:
                    table_text.append(cell)

    # Include the title, editable text and table content.
    # Deduplicate identical strings while preserving order.
    search_parts = []
    seen = set()

    for part in [title, *text_elements, *table_text]:
        part = clean_text(part)

        if part and part.casefold() not in seen:
            search_parts.append(part)
            seen.add(part.casefold())

    search_text = "\n".join(search_parts)

    return {
        "source_file": source_file,
        "presentation_title": deck_title,
        "slide_number": slide_number,
        "slide_title": title,
        "slide_title_method": title_method,
        "layout_name": slide.slide_layout.name,
        "text_elements": text_elements,
        "tables": tables,
        "images": images,
        "notes": notes,
        "search_text": search_text,
    }


# --------------------------------------------------
# 5. PROCESS ALL POWERPOINT FILES
# --------------------------------------------------

def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    RECORDS_FILE.parent.mkdir(parents=True, exist_ok=True)

    ppt_files = sorted(
        path
        for path in INPUT_DIR.glob("*.pptx")
        if not path.name.startswith("~$")
    )

    print(f"Found {len(ppt_files)} PowerPoint files.")

    if not ppt_files:
        print("No .pptx files found in input_ppts.")
        return

    all_slides = []
    processed_files = 0

    for ppt_path in ppt_files:
        print(f"\nProcessing: {ppt_path.name}")

        try:
            presentation = Presentation(str(ppt_path))
        except Exception as error:
            print(f"Could not open {ppt_path.name}: {error}")
            continue

        processed_files += 1
        deck_title = get_presentation_title(presentation)

        print(f"Deck title: {deck_title[:100]}")

        # IMPORTANT:
        # slide_height belongs to Presentation, not Slide.
        slide_height = presentation.slide_height

        for number, slide in enumerate(
            presentation.slides, start=1
        ):
            record = extract_slide(
                slide=slide,
                source_file=ppt_path.name,
                slide_number=number,
                deck_title=deck_title,
                slide_height=slide_height,
            )

            all_slides.append(record)

            # Display inferred titles for manual verification.
            if record["slide_title_method"] != "title_placeholder":
                print(
                    f"  Slide {number}: "
                    f"{record['slide_title'][:90]!r} "
                    f"[{record['slide_title_method']}]"
                )

    # Save records only after processing all readable files.
    with RECORDS_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            all_slides,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 55)
    print(f"Successfully opened presentations: {processed_files}")
    print(f"Total slide records: {len(all_slides)}")
    print(f"Saved to: {RECORDS_FILE}")
    print("=" * 55)

    print(
        "\nNext step: inspect the inferred titles before "
        "rebuilding image records and embeddings."
    )


if __name__ == "__main__":
    main()
