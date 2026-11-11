# STEP 4 - Main presentation-generation program.

# Retrieves source slides, separates text and image-only evidence,

# generates text slides with Gemini, and inserts relevant original images.



import json

import math

import re

from io import BytesIO

from pathlib import Path



from PIL import Image

from pptx import Presentation

from pptx.util import Inches, Pt



from config import OUTPUT_DIR

from generate_content import generate_slide_content

from hybrid_search import hybrid_search, slides

from evidence_selection import select_evidence



# Reuse the existing image-search implementation.

from search_images import (

    image_search_detailed,

    records as image_records,

)





# --------------------------------------------------

# CONFIGURATION

# --------------------------------------------------



MIN_SOURCE_SLIDES = 15

RETRIEVAL_MULTIPLIER = 3

IMAGE_CANDIDATES = 10



IMAGE_LEFT = 7.25

IMAGE_TOP = 1.75

IMAGE_BOX_WIDTH = 5.35

IMAGE_BOX_HEIGHT = 4.65



AVOID_DUPLICATE_IMAGES = True



PRESENTATION_WIDTH = 13.333

PRESENTATION_HEIGHT = 7.5





# --------------------------------------------------

# 1. RETRIEVAL

# --------------------------------------------------



def calculate_retrieval_count(requested_slides, total_slides):

    candidate_count = max(

        MIN_SOURCE_SLIDES,

        requested_slides * RETRIEVAL_MULTIPLIER,

    )

    return min(candidate_count, total_slides)





def retrieve_slides(query, top_k=MIN_SOURCE_SLIDES):

    """Retrieve ranked source slides and assign unique source IDs."""

    retrieved = []



    for index, score in hybrid_search(query, top_k=top_k):

        record = dict(slides[index])

        record["retrieval_score"] = float(score)

        record["_slide_index"] = index

        retrieved.append(record)



    for index, record in enumerate(retrieved, start=1):

        record["source_id"] = f"source_{index:03d}"



    return retrieved





# --------------------------------------------------

# 2. GENERAL HELPERS

# --------------------------------------------------



def safe_filename(text):

    name = re.sub(

        r"[^A-Za-z0-9]+",

        "_",

        str(text),

    ).strip("_")

    return (name or "presentation")[:60]





def get_image_path(record):

    """Resolve an image record's path within the output directory."""

    raw_path = Path(record.get("image_path", ""))



    if raw_path.is_absolute():

        return raw_path



    return OUTPUT_DIR / raw_path





def normalize_text(text):

    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()





def collect_text_values(value):

    """

    Collect actual text values from text-element/table structures.

    Numeric formatting and shape metadata are ignored.

    """

    collected = []



    if isinstance(value, str):

        if value.strip():

            collected.append(value.strip())



    elif isinstance(value, (list, tuple)):

        for item in value:

            collected.extend(collect_text_values(item))



    elif isinstance(value, dict):

        preferred_keys = {

            "text", "content", "value", "cell_text",

            "paragraphs", "rows", "cells", "data",

        }



        for key, item in value.items():

            if str(key).casefold() in preferred_keys:

                collected.extend(collect_text_values(item))



    return collected





def remove_title_only_lines(text, title):

    """Remove lines that contain only the slide title."""

    title_normalized = normalize_text(title)

    remaining = []



    for line in str(text or "").splitlines():

        line = line.strip()



        if not line:

            continue



        if title_normalized and normalize_text(line) == title_normalized:

            continue



        remaining.append(line)



    return "\n".join(remaining).strip()





def get_meaningful_text(source):

    """

    Return explanatory text excluding the slide title.



    A slide containing only its title and an image is not treated as

    text-bearing evidence.

    """

    title = source.get("slide_title", "")



    text_parts = []



    # Editable text elements, excluding the title itself.

    text_parts.extend(

        remove_title_only_lines(value, title)

        for value in collect_text_values(source.get("text_elements", []))

    )



    # Table content is meaningful source material.

    text_parts.extend(

        collect_text_values(source.get("tables", []))

    )



    # Speaker notes may contain useful explanatory information.

    for key in ("notes", "speaker_notes", "slide_notes"):

        value = source.get(key, "")

        if isinstance(value, str):

            text_parts.append(remove_title_only_lines(value, title))

        else:

            text_parts.extend(collect_text_values(value))



    # Compatibility with records that store extracted body text in

    # native_search_text or content instead of text_elements.

    if not any(part.strip() for part in text_parts):

        for key in ("native_search_text", "content"):

            value = source.get(key, "")

            if isinstance(value, str):

                text_parts.append(remove_title_only_lines(value, title))



    cleaned = []

    seen = set()



    for part in text_parts:

        part = str(part or "").strip()

        normalized = normalize_text(part)



        if normalized and normalized not in seen:

            cleaned.append(part)

            seen.add(normalized)



    return "\n".join(cleaned).strip()





def source_slide_key(record):

    """Stable key connecting a slide record to its extracted images."""

    return (

        str(record.get("source_file", record.get("document_name", ""))),

        str(

            record.get(

                "slide_number",

                record.get("original_slide_number", ""),

            )

        ),

    )





def get_images_for_source(source):

    """Return all extracted image records belonging to one source slide."""

    key = source_slide_key(source)



    return [

        record

        for record in image_records

        if source_slide_key(record) == key

    ]





GENERIC_QUERY_WORDS = {
    "explain", "describe", "discuss", "what", "how", "why",
    "and", "or", "the", "a", "an", "of", "for", "in", "on",
    "with", "its", "their", "this", "that", "from", "across",
    "covering", "include", "including", "summary", "presentation",
    "main", "idea", "ideas", "strength", "strengths", "limitation",
    "limitations", "suitable", "task", "tasks", "application",
    "applications", "advantage", "advantages", "disadvantage",
    "disadvantages", "feature", "features", "overview", "introduction",
    "architecture", "working", "implementation", "method", "methods",
    "machine", "learning", "algorithm", "algorithms", "model", "models",
    "technique", "techniques", "system", "systems",
}

# Expand common abbreviations so a query such as "SOM" can match a
# source slide titled "Self-Organizing Maps", and vice versa.
TOPIC_ALIASES = {
    "som": {"som", "soms", "self", "organizing", "organization", "maps"},
    "soms": {"som", "soms", "self", "organizing", "organization", "maps"},
    "vae": {"vae", "vaes", "variational", "autoencoder", "autoencoders"},
    "vaes": {"vae", "vaes", "variational", "autoencoder", "autoencoders"},
    "gan": {"gan", "gans", "generative", "adversarial"},
    "gans": {"gan", "gans", "generative", "adversarial"},
    "cnn": {"cnn", "cnns", "convolutional"},
    "rnn": {"rnn", "rnns", "recurrent"},
    "lstm": {"lstm", "lstms", "long", "short", "term", "memory"},
}


def get_topic_tokens(text):
    """Extract topic-specific words, excluding generic request language."""
    tokens = set(re.findall(r"[a-z0-9]+", str(text).lower()))
    tokens -= GENERIC_QUERY_WORDS

    expanded = set(tokens)
    for token in list(tokens):
        expanded.update(TOPIC_ALIASES.get(token, set()))

    # Also recognize expanded names when the acronym appears in the query.
    normalized = " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))
    phrase_aliases = {
        "self organizing map": {"som", "soms", "self", "organizing", "maps"},
        "self organizing maps": {"som", "soms", "self", "organizing", "maps"},
        "variational autoencoder": {"vae", "vaes", "variational", "autoencoder"},
        "generative adversarial network": {"gan", "gans", "generative", "adversarial"},
        "convolutional neural network": {"cnn", "cnns", "convolutional"},
        "recurrent neural network": {"rnn", "rnns", "recurrent"},
    }
    for phrase, aliases in phrase_aliases.items():
        if phrase in normalized:
            expanded.update(aliases)

    return expanded


def image_is_relevant(result, query, image_record):
    """
    Reject images matched only by generic terms such as "architecture".
    Require topic overlap plus a meaningful retrieval score.
    """
    query_topics = get_topic_tokens(query)
    candidate_text = " ".join([
        str(image_record.get("slide_title", "")),
        str(image_record.get("context_text", "")),
    ])
    candidate_topics = get_topic_tokens(candidate_text)

    if not query_topics or not query_topics.intersection(candidate_topics):
        return False

    return (
        result.get("exact_title_match", False)
        or result.get("exact_context_match", False)
        or result.get("semantic_score", 0) >= 0.40
        or result.get("keyword_score", 0) >= 8
    )


def get_bullet_texts(slide_data):

    """Extract bullet text from Gemini's JSON structure."""

    bullets = slide_data.get("bullets", [])

    texts = []



    for bullet in bullets:

        if isinstance(bullet, dict):

            text = str(bullet.get("text", "")).strip()

            if text:

                texts.append(text)



        elif isinstance(bullet, str) and bullet.strip():

            texts.append(bullet.strip())



    return texts





# --------------------------------------------------

# 3. CLASSIFY EVIDENCE AND FIND RELEVANT ORIGINAL IMAGES

# --------------------------------------------------



def classify_selected_evidence(selected_evidence):

    """

    Separate evidence into:

      1. Text-bearing slides: sent to Gemini.

      2. Image-only slides: eligible for direct image insertion.

      3. Empty slides: ignored.

    """

    text_sources = []

    image_only_sources = []

    empty_sources = []



    for source in selected_evidence:

        meaningful_text = get_meaningful_text(source)

        source_images = get_images_for_source(source)



        if meaningful_text:

            text_sources.append(source)



        elif source_images:

            image_only_sources.append(source)



        else:

            empty_sources.append(source)



    return text_sources, image_only_sources, empty_sources





def retrieve_relevant_image_only_slides(

    query,

    image_only_sources,

    requested_slide_count,

):

    """

    Search image records using the user's query, then retain only images

    belonging to selected image-only source slides.



    The image search uses title-weighted keyword matching, BM25,

    MiniLM semantic similarity, and RRF.

    """

    if not image_only_sources or not image_records:

        return []



    selected_keys = {

        source_slide_key(source)

        for source in image_only_sources

    }



    # Ask for all ranked image records so a relevant image on a selected

    # source slide is not missed merely because it falls below TOP_K=10.

    search_results = image_search_detailed(

        query,

        top_k=len(image_records),

    )



    relevant_by_slide = {}



    for result in search_results:

        image_index = result.get("index")



        if image_index is None or not 0 <= image_index < len(image_records):

            continue



        image_record = image_records[image_index]



        if not image_is_relevant(result, query, image_record):

            continue

        image_index = result.get("index")



        if image_index is None or not 0 <= image_index < len(image_records):

            continue



        image_record = image_records[image_index]

        key = source_slide_key(image_record)



        if key not in selected_keys:

            continue



        image_path = get_image_path(image_record)



        if not image_path.is_file():

            print(f"  Skipping missing original image: {image_path}")

            continue



        relevant_by_slide.setdefault(key, []).append({

            "record": image_record,

            "path": image_path,

            "score": float(result.get("final_score", 0.0)),

            "search_result": result,

        })



    direct_slides = []

    used_hashes = set()



    # Preserve MMR evidence order.

    for source in image_only_sources:

        key = source_slide_key(source)

        matches = relevant_by_slide.get(key, [])



        if not matches:

            continue



        matches.sort(

            key=lambda item: item["score"],

            reverse=True,

        )



        slide_images = []



        for match in matches:

            image_hash = match["record"].get("image_hash", "")



            if (

                AVOID_DUPLICATE_IMAGES

                and image_hash

                and image_hash in used_hashes

            ):

                continue



            if image_hash:

                used_hashes.add(image_hash)



            slide_images.append(match)



        if slide_images:

            direct_slides.append({

                "source": source,

                "images": slide_images,

            })



        if len(direct_slides) >= requested_slide_count:

            break



    return direct_slides





# --------------------------------------------------

# 4. EXISTING IMAGE RETRIEVAL FOR GEMINI-GENERATED SLIDES

# --------------------------------------------------



def retrieve_image_for_slide(slide_data, used_hashes):

    """

    Search by generated slide title first. If no exact title match is

    found, search by title plus bullet text. Avoid duplicate images.

    """

    title = str(slide_data.get("title", "")).strip()

    bullet_texts = get_bullet_texts(slide_data)



    if not title:

        return None



    title_results = image_search_detailed(

        title,

        top_k=IMAGE_CANDIDATES,

    )



    exact_title_results = [

        result

        for result in title_results

        if result.get("exact_title_match", False)

    ]



    if exact_title_results:

        candidates = exact_title_results

    else:

        combined_query = " ".join([title] + bullet_texts[:3])

        candidates = image_search_detailed(

            combined_query,

            top_k=IMAGE_CANDIDATES,

        )



    for result in candidates:

        image_index = result.get("index")



        if image_index is None or not 0 <= image_index < len(image_records):

            continue



        image_record = image_records[image_index]



        image_query = " ".join([title] + bullet_texts[:3])



        if not image_is_relevant(result, image_query, image_record):

            continue

        image_index = result.get("index")



        if image_index is None or not 0 <= image_index < len(image_records):

            continue



        image_record = image_records[image_index]

        image_path = get_image_path(image_record)



        if not image_path.is_file():

            print(f"  Skipping missing image: {image_path}")

            continue



        image_hash = image_record.get("image_hash", "")



        if (

            AVOID_DUPLICATE_IMAGES

            and image_hash

            and image_hash in used_hashes

        ):

            continue



        if image_hash:

            used_hashes.add(image_hash)



        return {

            "path": image_path,

            "record": image_record,

            "score": result.get("final_score", 0),

            "search_result": result,

        }



    return None





# --------------------------------------------------

# 5. POWERPOINT LAYOUT HELPERS

# --------------------------------------------------



def add_textbox(

    slide,

    left,

    top,

    width,

    height,

    text="",

    font_size=20,

    bold=False,

):

    """Create a formatted text box."""

    shape = slide.shapes.add_textbox(

        Inches(left),

        Inches(top),

        Inches(width),

        Inches(height),

    )



    text_frame = shape.text_frame

    text_frame.word_wrap = True

    text_frame.margin_left = Inches(0.05)

    text_frame.margin_right = Inches(0.05)

    text_frame.margin_top = Inches(0.05)

    text_frame.margin_bottom = Inches(0.05)



    if text:

        paragraph = text_frame.paragraphs[0]

        paragraph.text = str(text)

        paragraph.font.size = Pt(font_size)

        paragraph.font.bold = bold



    return shape





def get_image_dimensions(image_path, record):

    width = float(record.get("width", 0) or 0)

    height = float(record.get("height", 0) or 0)



    if width > 0 and height > 0:

        return width, height



    with Image.open(image_path) as image:

        return image.size





def add_image_in_box(slide, image_path, record, left, top, box_width, box_height):

    """Insert an image from bytes and preserve its aspect ratio."""

    try:

        image_width, image_height = get_image_dimensions(

            image_path,

            record,

        )



        if image_width <= 0 or image_height <= 0:

            return False



        aspect_ratio = image_width / image_height



        fitted_width = min(

            box_width,

            box_height * aspect_ratio,

        )

        fitted_height = fitted_width / aspect_ratio



        fitted_left = left + (box_width - fitted_width) / 2

        fitted_top = top + (box_height - fitted_height) / 2



        # Use BytesIO because image_store files use a .bin extension.

        image_stream = BytesIO(image_path.read_bytes())



        slide.shapes.add_picture(

            image_stream,

            Inches(fitted_left),

            Inches(fitted_top),

            width=Inches(fitted_width),

            height=Inches(fitted_height),

        )



        return True



    except Exception as error:

        print(f"  Could not insert image {image_path}: {error}")

        return False





def add_fitted_picture(slide, image_path, record):

    """Preserve the existing right-side image layout for Gemini slides."""

    return add_image_in_box(

        slide=slide,

        image_path=image_path,

        record=record,

        left=IMAGE_LEFT,

        top=IMAGE_TOP,

        box_width=IMAGE_BOX_WIDTH,

        box_height=IMAGE_BOX_HEIGHT,

    )





def format_bullets(text_frame, bullet_texts, font_size=18):

    """Populate a PowerPoint text frame with generated bullet points."""

    text_frame.clear()

    text_frame.word_wrap = True

    text_frame.margin_left = Inches(0.08)

    text_frame.margin_right = Inches(0.08)

    text_frame.margin_top = Inches(0.05)

    text_frame.margin_bottom = Inches(0.05)



    for index, text in enumerate(bullet_texts):

        paragraph = (

            text_frame.paragraphs[0]

            if index == 0

            else text_frame.add_paragraph()

        )



        paragraph.text = text

        paragraph.level = 0

        paragraph.font.size = Pt(font_size)

        paragraph.space_after = Pt(12)





# --------------------------------------------------

# 6. ADD DIRECT IMAGE-ONLY SOURCE SLIDES

# --------------------------------------------------



def add_direct_image_slide(

    presentation,

    direct_slide_data,

    slide_number,

    presentation_title,

):

    """

    Add one slide using the original title and original image(s)

    extracted from the matching source PowerPoint slide.

    """

    source = direct_slide_data["source"]

    images = direct_slide_data["images"]



    slide = presentation.slides.add_slide(

        presentation.slide_layouts[6]

    )



    source_title = str(

        source.get("slide_title", "Untitled slide")

    ).strip() or "Untitled slide"



    add_textbox(

        slide,

        left=0.7,

        top=0.12,

        width=11.9,

        height=0.28,

        text=presentation_title,

        font_size=10,

    )



    add_textbox(

        slide,

        left=0.7,

        top=0.48,

        width=11.9,

        height=0.85,

        text=source_title,

        font_size=29,

        bold=True,

    )



    # Arrange multiple original images in a simple grid.

    content_left = 0.7

    content_top = 1.55

    content_width = 11.95

    content_height = 5.45

    gap = 0.20



    image_count = len(images)

    columns = math.ceil(math.sqrt(image_count))

    rows = math.ceil(image_count / columns)



    cell_width = (content_width - gap * (columns - 1)) / columns

    cell_height = (content_height - gap * (rows - 1)) / rows



    inserted = 0



    for index, image_item in enumerate(images):

        row = index // columns

        column = index % columns



        left = content_left + column * (cell_width + gap)

        top = content_top + row * (cell_height + gap)



        if add_image_in_box(

            slide,

            image_item["path"],

            image_item["record"],

            left,

            top,

            cell_width,

            cell_height,

        ):

            inserted += 1



    # Preserve original-source provenance in speaker notes.

    notes = [

        "Directly inserted original image-only source slide.",

        f"Source presentation: {source.get('source_file', '')}",

        f"Original slide number: {source.get('slide_number', '')}",

        f"Original slide title: {source_title}",

        f"Source ID: {source.get('source_id', '')}",

        "Original images:",

    ]



    for image_item in images:

        image_record = image_item["record"]

        notes.append(

            f"- {image_record.get('image_id', '')}; "

            f"image hash: {image_record.get('image_hash', '')}; "

            f"search score: {image_item['score']:.6f}"

        )



    slide.notes_slide.notes_text_frame.text = "\n".join(notes)



    print(

        f"Direct image slide {slide_number}: {source_title} "

        f"({inserted}/{image_count} images inserted)"

    )



    return inserted





# --------------------------------------------------

# 7. CREATE THE COMBINED POWERPOINT

# --------------------------------------------------



def create_powerpoint(

    content,

    output_path,

    retrieved_sources,

    direct_image_slides=None,

    requested_slide_count=None,

):

    """

    Create Gemini-generated slides followed by relevant original

    image-only slides. No additional cover slide is added.

    """

    direct_image_slides = direct_image_slides or []



    presentation = Presentation()

    presentation.slide_width = Inches(PRESENTATION_WIDTH)

    presentation.slide_height = Inches(PRESENTATION_HEIGHT)



    presentation_title = content.get(

        "presentation_title",

        "Presentation",

    )



    presentation.core_properties.title = presentation_title

    presentation.core_properties.subject = (

        "Source-grounded presentation generated by PresentationMaker"

    )



    source_lookup = {

        str(source.get("source_id")): source

        for source in retrieved_sources

    }



    # Prevent generated-slide image search from reusing images that

    # are already inserted as direct image-only slides.

    used_hashes = set()



    for direct_item in direct_image_slides:

        for image_item in direct_item["images"]:

            image_hash = image_item["record"].get("image_hash", "")

            if image_hash:

                used_hashes.add(image_hash)



    inserted_images = 0

    generated_slides = content.get("slides", [])



    # First create Gemini-generated slides.

    for slide_number, slide_data in enumerate(

        generated_slides,

        start=1,

    ):

        slide = presentation.slides.add_slide(

            presentation.slide_layouts[6]

        )



        title = str(

            slide_data.get("title", "Untitled slide")

        )

        bullet_texts = get_bullet_texts(slide_data)



        print(f"\nGenerated slide {slide_number}: {title}")



        selected_image = retrieve_image_for_slide(

            slide_data,

            used_hashes,

        )



        has_image = selected_image is not None



        if has_image:

            image_record = selected_image["record"]

            print(

                "  Selected supporting image from "

                f"{image_record.get('source_file', '')}, "

                f"slide {image_record.get('slide_number', '')}"

            )

        else:

            print("  No sufficiently relevant supporting image selected.")



        add_textbox(

            slide,

            left=0.7,

            top=0.12,

            width=11.9,

            height=0.28,

            text=presentation_title,

            font_size=10,

        )



        add_textbox(

            slide,

            left=0.7,

            top=0.48,

            width=11.9,

            height=0.85,

            text=title,

            font_size=29,

            bold=True,

        )



        body_width = 6.15 if has_image else 11.9



        body_shape = slide.shapes.add_textbox(

            Inches(0.75),

            Inches(1.55),

            Inches(body_width),

            Inches(5.35),

        )



        format_bullets(

            body_shape.text_frame,

            bullet_texts,

            font_size=18,

        )



        image_inserted = False



        if has_image:

            image_inserted = add_fitted_picture(

                slide,

                selected_image["path"],

                selected_image["record"],

            )



            if image_inserted:

                inserted_images += 1



        # Add source provenance to speaker notes.

        notes = []

        slide_source_ids = slide_data.get("source_ids", [])



        if slide_source_ids:

            notes.append("Content source slides:")



            for source_id in slide_source_ids:

                source = source_lookup.get(str(source_id), {})



                source_file = source.get(

                    "source_file",

                    source.get("document_name", ""),

                )



                source_slide_number = source.get(

                    "slide_number",

                    source.get("original_slide_number", ""),

                )



                notes.append(

                    f"- [{source_id}] {source_file}, "

                    f"slide {source_slide_number}"

                )



        for bullet_index, bullet in enumerate(

            slide_data.get("bullets", []),

            start=1,

        ):

            if not isinstance(bullet, dict):

                continue



            bullet_sources = bullet.get("source_ids", [])



            if bullet_sources:

                notes.append(

                    f"Bullet {bullet_index} source IDs: "

                    + ", ".join(map(str, bullet_sources))

                )



        if selected_image and image_inserted:

            image_record = selected_image["record"]

            notes.extend([

                "",

                "Selected supporting image source:",

                f"- Presentation: {image_record.get('source_file', '')}",

                f"- Slide: {image_record.get('slide_number', '')}",

                f"- Image hash: {image_record.get('image_hash', '')}",

                f"- Image search score: {selected_image['score']:.6f}",

            ])



        if notes:

            slide.notes_slide.notes_text_frame.text = "\n".join(notes)



    # Then add exact original image-only source slides.

    for direct_item in direct_image_slides:

        inserted_images += add_direct_image_slide(

            presentation,

            direct_item,

            slide_number=len(presentation.slides) + 1,

            presentation_title=presentation_title,

        )



    final_count = len(presentation.slides)



    if (
        requested_slide_count is not None
        and final_count > requested_slide_count
    ):
        raise ValueError(
            f"Slide limit exceeded: maximum "
            f"{requested_slide_count}, created {final_count}. "
            "The presentation was not saved."
        )



    presentation.save(output_path)



    print(f"\nImages inserted: {inserted_images}")

    print(f"PowerPoint slides created: {final_count}")



    return final_count





# --------------------------------------------------

# 8. MAIN PROGRAM

# --------------------------------------------------


def main(
    user_request=None,
    slide_count=None,
    target_audience="General audience",
    presentation_objective="Explain the requested topic clearly",
):
    print("\n=== PresentationMaker ===")

    # Support both Streamlit arguments and interactive terminal use.
    if user_request is None:
        user_request = input(
            "What presentation would you like to create? "
        ).strip()

    if not user_request:
        raise ValueError("Please enter a presentation request.")

    if slide_count is None:
        try:
            slide_count = int(
                input("How many slides do you need? ").strip()
            )
        except ValueError:
            raise ValueError("Please enter a valid whole number.")

    if not 1 <= slide_count <= 20:
        raise ValueError(
            "For this prototype, choose between 1 and 20 slides."
        )

    if not target_audience:
        target_audience = "General audience"

    if not presentation_objective:
        presentation_objective = "Explain the requested topic clearly"

    print("\nSearching your knowledge base...")

    # --------------------------------------------------
    # 1. RETRIEVE SOURCE SLIDES
    # --------------------------------------------------

    retrieval_count = calculate_retrieval_count(
        requested_slides=slide_count,
        total_slides=len(slides),
    )

    print(
        f"Retrieving up to {retrieval_count} candidate source slides..."
    )

    retrieved = retrieve_slides(
        user_request,
        top_k=retrieval_count,
    )

    if not retrieved:
        print("No relevant slides were found.")
        return None

    # --------------------------------------------------
    # 2. SELECT DIVERSE EVIDENCE USING MMR
    # --------------------------------------------------

    evidence_count = min(
        len(retrieved),
        slide_count * 2,
    )

    selected_evidence = select_evidence(
        candidates=retrieved,
        top_k=evidence_count,
    )

    if not selected_evidence:
        print("No sufficient evidence selected.")
        return None

    print(
        f"\nSelected {len(selected_evidence)} evidence slides "
        f"from {len(retrieved)} candidates."
    )

    # Save retrieval and selected evidence for auditing.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    retrieval_path = OUTPUT_DIR / "last_retrieval.json"

    with retrieval_path.open("w", encoding="utf-8") as file:
        json.dump(
            retrieved,
            file,
            indent=2,
            ensure_ascii=False,
        )

    selected_path = OUTPUT_DIR / "last_selected_evidence.json"

    with selected_path.open("w", encoding="utf-8") as file:
        json.dump(
            selected_evidence,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------
    # 3. CLASSIFY SELECTED EVIDENCE
    # --------------------------------------------------

    text_sources, image_only_sources, empty_sources = (
        classify_selected_evidence(selected_evidence)
    )

    print("\nEvidence classification:")
    print(f"  Text-bearing slides: {len(text_sources)}")
    print(f"  Image-only slides:   {len(image_only_sources)}")
    print(f"  Empty slides:        {len(empty_sources)}")

    if empty_sources:
        for source in empty_sources:
            print(
                "  Ignored empty slide: "
                f"{source.get('source_file', '')}, "
                f"slide {source.get('slide_number', '')}"
            )

    # --------------------------------------------------
    # 4. RETRIEVE RELEVANT ORIGINAL IMAGE-ONLY SLIDES
    # --------------------------------------------------

    direct_image_slides = retrieve_relevant_image_only_slides(
        query=user_request,
        image_only_sources=image_only_sources,
        requested_slide_count=slide_count,
    )

    image_slide_count = len(direct_image_slides)

    # Allocate the remaining slide budget to Gemini.
    text_slide_budget = max(
        0,
        slide_count - image_slide_count,
    )

    print("\nSlide allocation:")
    print(f"  Requested final slides: {slide_count}")
    print(f"  Relevant original image slides: {image_slide_count}")
    print(f"  Slides allocated to Gemini: {text_slide_budget}")

    for item in direct_image_slides:
        source = item["source"]

        print(
            f"  Direct image slide: "
            f"{source.get('slide_title', '')} "
            f"({len(item['images'])} image(s))"
        )

    # --------------------------------------------------
    # 5. GENERATE TEXT CONTENT ONLY WHEN AVAILABLE
    # --------------------------------------------------

    if text_slide_budget > 0 and text_sources:

        print(
            f"\nGenerating {text_slide_budget} text slides with Gemini..."
        )

        content = generate_slide_content(
            user_request=user_request,
            retrieved_slides=text_sources,
            slide_count=text_slide_budget,
            target_audience=target_audience,
            presentation_objective=presentation_objective,
        )

        if content.get("status") == "insufficient_source_material":

            print(
                "\nWARNING: Gemini reported insufficient source material."
            )

            for gap in content.get("content_gaps", []):
                print(
                    f"- Gap: {gap.get('topic', '')}; "
                    f"needed: {gap.get('needed_material', '')}"
                )

    else:

        # IMPORTANT:
        # Do not call Gemini when only image evidence is available.
        # Original images will be inserted directly into the PPT.

        if direct_image_slides:
            print(
                "\nNo text-bearing evidence is available, or the "
                "text-slide budget is zero."
            )
            print("Skipping Gemini and using original image slides.")

        content = {
            "presentation_title": safe_filename(
                user_request
            ).replace("_", " "),
            "requested_slide_count": slide_count,
            "generated_slide_count": 0,
            "status": "image_only",
            "content_gaps": [],
            "slides": [],
        }

    # --------------------------------------------------
    # 6. VALIDATE AVAILABLE EVIDENCE
    # --------------------------------------------------

    if not content.get("slides") and not direct_image_slides:

        print(
            "\nNo relevant text-bearing or image-only evidence was found. "
            "No presentation can be created."
        )

        return None

    generated_slides = content.get("slides", [])

    # Never exceed the number of slides allocated to Gemini.
    if len(generated_slides) > text_slide_budget:

        generated_slides = generated_slides[:text_slide_budget]

        content["slides"] = generated_slides

    generated_count = len(generated_slides)

    final_expected_count = (
        generated_count + image_slide_count
    )

    content["requested_slide_count"] = slide_count
    content["generated_slide_count"] = generated_count
    content["direct_image_slide_count"] = image_slide_count
    content["final_slide_count"] = final_expected_count

    # Save generated content, including image-only status.
    content_path = OUTPUT_DIR / "generated_content.json"

    with content_path.open("w", encoding="utf-8") as file:
        json.dump(
            content,
            file,
            indent=2,
            ensure_ascii=False,
        )

    if final_expected_count > slide_count:
        raise ValueError(
            f"Slide limit exceeded: requested maximum {slide_count}, "
            f"but allocation produced {final_expected_count}."
        )

    if final_expected_count == 0:
        print("\nNo slides could be generated.")
        return None

    print(
        f"\nEvidence supports {final_expected_count} of "
        f"{slide_count} requested maximum slides."
    )

    # --------------------------------------------------
    # 7. CREATE FINAL POWERPOINT
    # --------------------------------------------------

    output_path = OUTPUT_DIR / (
        f"{safe_filename(content.get('presentation_title', user_request))}.pptx"
    )

    print("\nCreating the combined PowerPoint...")

    final_count = create_powerpoint(
        content=content,
        output_path=output_path,
        retrieved_sources=selected_evidence,
        direct_image_slides=direct_image_slides,
        requested_slide_count=slide_count,
    )

    print("\nPresentation created successfully!")
    print(f"PowerPoint:          {output_path}")
    print(f"Generated content:   {content_path}")
    print(f"Retrieved evidence:   {retrieval_path}")
    print(f"Selected evidence:   {selected_path}")
    print(f"Requested slides:    {slide_count}")
    print(f"Gemini slides:       {generated_count}")
    print(f"Original image slides: {image_slide_count}")
    print(f"Final slides:        {final_count}")

    return output_path


# --------------------------------------------------
# COMMAND-LINE ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--query", default=None)
    parser.add_argument("--slides", type=int, default=None)
    parser.add_argument(
        "--audience",
        default="General audience",
    )
    parser.add_argument(
        "--objective",
        default="Explain the requested topic clearly",
    )

    args = parser.parse_args()

    result = main(
        user_request=args.query,
        slide_count=args.slides,
        target_audience=args.audience,
        presentation_objective=args.objective,
    )

    if result:
        print(f"\nOUTPUT_FILE={result}")
