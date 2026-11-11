
import hashlib
import json
from io import BytesIO
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from config import INPUT_DIR, OUTPUT_DIR, RECORDS_FILE


IMAGE_STORE_DIR = OUTPUT_DIR / "image_store"
IMAGE_RECORDS_FILE = OUTPUT_DIR / "image_records.json"


def iter_picture_shapes(shapes):
    """Find pictures, including pictures inside grouped shapes."""
    for shape in shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from iter_picture_shapes(shape.shapes)
        elif shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            yield shape


def main():
    if not RECORDS_FILE.exists():
        raise FileNotFoundError(
            f"Slide records not found: {RECORDS_FILE}"
        )

    with open(RECORDS_FILE, "r", encoding="utf-8") as f:
        slide_records = json.load(f)

    # Match slide metadata by source presentation and slide number.
    slide_lookup = {
        (record["source_file"], int(record["slide_number"])): record
        for record in slide_records
    }

    IMAGE_STORE_DIR.mkdir(parents=True, exist_ok=True)

    image_records = []
    exact_hashes_seen = set()
    occurrence_count = 0
    unique_image_count = 0
    unmatched_slides = 0

    for ppt_path in sorted(INPUT_DIR.glob("*.pptx")):
        if ppt_path.name.startswith("~$"):
            continue

        print(f"Processing: {ppt_path.name}")

        try:
            prs = Presentation(str(ppt_path))
        except Exception as exc:
            print(f"  Could not open presentation: {exc}")
            continue

        for slide_number, slide in enumerate(prs.slides, start=1):
            key = (ppt_path.name, slide_number)
            slide_record = slide_lookup.get(key)

            if slide_record is None:
                unmatched_slides += 1
                continue

            for shape_index, shape in enumerate(
                iter_picture_shapes(slide.shapes), start=1
            ):
                occurrence_count += 1

                try:
                    image_bytes = shape.image.blob
                    image_hash = hashlib.sha256(
                        image_bytes
                    ).hexdigest()

                    # Save each exact image only once.
                    image_path = IMAGE_STORE_DIR / f"{image_hash}.bin"

                    if image_hash not in exact_hashes_seen:
                        image_path.write_bytes(image_bytes)
                        exact_hashes_seen.add(image_hash)
                        unique_image_count += 1

                    try:
                        with Image.open(BytesIO(image_bytes)) as img:
                            width, height = img.size
                            image_format = img.format
                    except Exception:
                        width, height = None, None
                        image_format = None

                    native_text = slide_record.get(
                        "native_search_text", ""
                    )
                    ocr_text = slide_record.get("ocr_text", "")
                    slide_title = slide_record.get(
                        "slide_title", ""
                    )
                    presentation_title = slide_record.get(
                        "presentation_title", ""
                    )

                    context_text = "\n".join(
                        part for part in [
                            f"Presentation: {presentation_title}",
                            f"Slide title: {slide_title}",
                            f"Slide content: {native_text}",
                            f"Text detected in images: {ocr_text}",
                        ]
                        if part.strip()
                    )

                    image_records.append({
                        "image_id": (
                            f"{ppt_path.stem}_S{slide_number}"
                            f"_P{shape_index}"
                        ),
                        "source_file": ppt_path.name,
                        "slide_number": slide_number,
                        "slide_title": slide_title,
                        "image_hash": image_hash,
                        "image_path": str(
                            Path("image_store") /
                            f"{image_hash}.bin"
                        ),
                        "image_format": image_format,
                        "width": width,
                        "height": height,
                        "shape_name": shape.name,
                        "context_text": context_text,
                    })

                except Exception as exc:
                    print(
                        f"  Skipping picture on slide "
                        f"{slide_number}: {exc}"
                    )

    # Write to a separate file; do not overwrite slide_records.json.
    with open(
        IMAGE_RECORDS_FILE, "w", encoding="utf-8"
    ) as f:
        json.dump(image_records, f, indent=2, ensure_ascii=False)

    print("\nImage record build complete.")
    print(f"Image appearances: {occurrence_count}")
    print(f"Unique exact image hashes: {unique_image_count}")
    print(f"Image records written: {len(image_records)}")
    print(f"Slides without matching records: {unmatched_slides}")
    print(f"Image records: {IMAGE_RECORDS_FILE}")
    print(f"Image store: {IMAGE_STORE_DIR}")


if __name__ == "__main__":
    main()
