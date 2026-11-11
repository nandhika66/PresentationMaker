
import json
import numpy as np

from config import EMBEDDING_DIM, OUTPUT_DIR
from embedder import MiniLMEmbedder


IMAGE_RECORDS_FILE = OUTPUT_DIR / "image_records.json"
IMAGE_EMBEDDINGS_FILE = OUTPUT_DIR / "image_embeddings.npy"


def main():
    if not IMAGE_RECORDS_FILE.exists():
        raise FileNotFoundError(
            f"Image records not found: {IMAGE_RECORDS_FILE}"
        )

    with open(IMAGE_RECORDS_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not records:
        raise ValueError("No image records to embed.")

    # Each image is represented by its slide context.
    texts = [
        record.get("context_text", "").strip()
        or record.get("slide_title", "")
        for record in records
    ]

    model = MiniLMEmbedder()

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    if embeddings.shape != (len(records), EMBEDDING_DIM):
        raise ValueError(
            f"Unexpected embedding shape: {embeddings.shape}; "
            f"expected ({len(records)}, {EMBEDDING_DIM})"
        )

    np.save(IMAGE_EMBEDDINGS_FILE, embeddings)

    print("\nImage embeddings created successfully.")
    print(f"Image records: {len(records)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Saved to: {IMAGE_EMBEDDINGS_FILE}")


if __name__ == "__main__":
    main()
