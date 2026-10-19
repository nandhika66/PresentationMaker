# STEP 2 - Convert every slide record into a MiniLM embedding vector
# and save them to output/slide_embeddings.npy (the knowledge base).
# Run this again every time you re-run read_ppt.py.
# Runs MiniLM through onnxruntime (no PyTorch needed).

import json

import numpy as np

from config import EMBEDDING_DIM, EMBEDDING_MODEL, EMBEDDINGS_FILE, RECORDS_FILE
from embedder import MiniLMEmbedder


def build_text(slide):
    """Create exactly ONE text per slide."""
    return (
        f"Presentation: {slide.get('presentation_title', '')}\n"
        f"Slide title: {slide.get('slide_title', '')}\n"
        f"Slide content: {slide.get('search_text', '')}"
    )


def main():
    if not RECORDS_FILE.exists():
        raise FileNotFoundError(
            f"{RECORDS_FILE} not found. Run: python read_ppt.py"
        )

    with open(RECORDS_FILE, "r", encoding="utf-8") as file:
        slides = json.load(file)

    print(f"Loaded {len(slides)} slide records.")

    texts = [build_text(slide) for slide in slides]

    print(f"Loading model: {EMBEDDING_MODEL}")
    print("(The first run downloads about 90 MB, later runs use the cache.)")
    model = MiniLMEmbedder()

    print(f"Embedding {len(texts)} slides...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    if embeddings.shape != (len(slides), EMBEDDING_DIM):
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")

    np.save(EMBEDDINGS_FILE, embeddings)

    print("Embedding generation complete.")
    print(f"Vector shape: {embeddings.shape}")
    print(f"Saved to: {EMBEDDINGS_FILE}")
    print("\nNext step: python generate_presentation.py")


if __name__ == "__main__":
    main()
