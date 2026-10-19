# STEP 3 - Hybrid search: custom keyword + BM25 + MiniLM, combined with
# Reciprocal Rank Fusion (RRF).
# Run directly for an interactive search, or import hybrid_search().

import json
import re

import numpy as np
from rank_bm25 import BM25Okapi

from config import EMBEDDINGS_FILE, RECORDS_FILE


TOP_K = 5
RRF_K = 60

KEYWORD_WEIGHT = 2
BM25_WEIGHT = 1
SEMANTIC_WEIGHT = 1


# ---------- Load the knowledge base ----------

if not RECORDS_FILE.exists():
    raise FileNotFoundError(
        f"{RECORDS_FILE.name} not found. Run: python read_ppt.py"
    )

with open(RECORDS_FILE, "r", encoding="utf-8") as file:
    slides = json.load(file)


def load_semantic_search():
    """Load embeddings + MiniLM. Returns (None, None) if unavailable,
    so the search still works with keyword + BM25 only."""

    if not EMBEDDINGS_FILE.exists():
        print("WARNING: slide_embeddings.npy not found -> run: python build_embeddings.py")
        return None, None

    vectors = np.load(EMBEDDINGS_FILE)
    if vectors.shape[0] != len(slides):
        print(
            f"WARNING: {len(slides)} slides but {vectors.shape[0]} embeddings "
            "-> run: python build_embeddings.py"
        )
        return None, None

    try:
        from embedder import MiniLMEmbedder
        return vectors, MiniLMEmbedder()
    except Exception as error:
        print(f"WARNING: MiniLM could not be loaded ({type(error).__name__}: {error})")
        return None, None


embeddings, model = load_semantic_search()

if model is None:
    print("Semantic search is OFF - using keyword + BM25 only.\n")
else:
    print(f"Knowledge base: {len(slides)} slides, embeddings {embeddings.shape}")


# ---------- Tokenization ----------

def tokenize_text(text):
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def tokenize(text):
    """Return unique words for keyword matching."""
    return set(tokenize_text(text))


# ---------- Build BM25 index once ----------

slide_tokens = [
    tokenize_text(" ".join([
        slide.get("presentation_title", ""),
        slide.get("slide_title", ""),
        slide.get("search_text", ""),
    ]))
    for slide in slides
]

bm25 = BM25Okapi(slide_tokens)


# ---------- Keyword search (3 x title matches + content matches) ----------

def keyword_search(query):
    query_words = tokenize(query)
    results = []

    for index, slide in enumerate(slides):
        title_words = tokenize(
            slide.get("slide_title", "") + " " + slide.get("presentation_title", "")
        )
        content_words = tokenize(
            slide.get("search_text", "") + " " + slide.get("presentation_title", "")
        )

        score = 3 * len(query_words & title_words) + len(query_words & content_words)

        if score > 0:
            results.append((index, score))

    results.sort(key=lambda item: item[1], reverse=True)
    return results


# ---------- BM25 search ----------

def bm25_search(query):
    scores = bm25.get_scores(tokenize_text(query))

    results = [
        (index, float(score))
        for index, score in enumerate(scores)
        if score > 0
    ]

    results.sort(key=lambda item: item[1], reverse=True)
    return results


# ---------- Semantic search (cosine similarity) ----------

def semantic_search(query):
    if model is None:
        return []

    query_vector = model.encode(query, normalize_embeddings=True)

    # Normalized vectors -> dot product equals cosine similarity
    scores = embeddings @ query_vector
    ranked_indices = np.argsort(scores)[::-1]

    return [(int(index), float(scores[index])) for index in ranked_indices]


# ---------- Hybrid search using RRF ----------

def hybrid_search(query, top_k=TOP_K):
    hybrid_scores = {}

    rankings = [
        (keyword_search(query), KEYWORD_WEIGHT),
        (bm25_search(query), BM25_WEIGHT),
        (semantic_search(query), SEMANTIC_WEIGHT),
    ]

    for results, weight in rankings:
        for rank, (index, _) in enumerate(results, start=1):
            hybrid_scores[index] = hybrid_scores.get(index, 0) + weight / (RRF_K + rank)

    ranked = sorted(hybrid_scores.items(), key=lambda item: item[1], reverse=True)
    return ranked[:top_k]


# ---------- Interactive mode ----------

def display_results(results, score_label):
    for rank, (index, score) in enumerate(results, start=1):
        slide = slides[index]
        print(f"\nRank {rank} | {score_label}: {score:.4f}")
        print(f"Presentation: {slide.get('presentation_title', 'Unknown')[:80]}")
        print(f"Title: {slide.get('slide_title', '')}")
        print(f"Source: {slide.get('source_file', '')} (slide {slide.get('slide_number', '')})")
        print(f"Content: {slide.get('search_text', '')[:350]}")


def main():
    print("Hybrid search ready. Type 'exit' to stop.")

    while True:
        query = input("\nSearch topic: ").strip()

        if query.lower() == "exit":
            break
        if not query:
            continue

        display_results(hybrid_search(query), "Hybrid score")


if __name__ == "__main__":
    main()
