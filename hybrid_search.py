# hybrid_search.py
# Selected retrieval method:
# BM25 + MiniLM Semantic Search + Reciprocal Rank Fusion

import json
import re

import numpy as np
from rank_bm25 import BM25Okapi

from config import EMBEDDINGS_FILE, RECORDS_FILE
from embedder import MiniLMEmbedder


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

RRF_K = 60

BM25_WEIGHT = 1
SEMANTIC_WEIGHT = 1

DEFAULT_TOP_K = 5


# ---------------------------------------------------------
# LOAD KNOWLEDGE BASE
# ---------------------------------------------------------

if not RECORDS_FILE.exists():
    raise FileNotFoundError(
        f"{RECORDS_FILE} not found. Run: python read_ppt.py"
    )

with RECORDS_FILE.open("r", encoding="utf-8") as file:
    slides = json.load(file)

if not slides:
    raise ValueError("No slide records found.")

if not EMBEDDINGS_FILE.exists():
    raise FileNotFoundError(
        f"{EMBEDDINGS_FILE} not found. "
        "Run your MiniLM embedding-generation script first."
    )

embeddings = np.load(EMBEDDINGS_FILE)

if embeddings.shape[0] != len(slides):
    raise ValueError(
        "Slide and embedding counts do not match. "
        f"Slides: {len(slides)}, embeddings: {embeddings.shape[0]}"
    )

print(f"Loaded {len(slides)} slides.")
print(f"MiniLM embedding shape: {embeddings.shape}")


# ---------------------------------------------------------
# LOAD MINILM MODEL
# ---------------------------------------------------------

model = MiniLMEmbedder()


# ---------------------------------------------------------
# TOKENIZATION
# ---------------------------------------------------------

def tokenize_text(text):
    """Convert text into lowercase alphanumeric tokens."""
    return re.findall(r"\b[a-z0-9]+\b", str(text).lower())


# ---------------------------------------------------------
# BUILD BM25 INDEX
# ---------------------------------------------------------

slide_tokens = []

for slide in slides:
    searchable_text = " ".join([
        str(slide.get("presentation_title", "") or ""),
        str(slide.get("slide_title", "") or ""),
        str(slide.get("search_text", "") or ""),
    ])

    slide_tokens.append(tokenize_text(searchable_text))

bm25 = BM25Okapi(slide_tokens)


# ---------------------------------------------------------
# BM25 SEARCH
# ---------------------------------------------------------

def bm25_search(query):
    """Return slides ranked by BM25 relevance."""

    query_tokens = tokenize_text(query)

    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)

    results = [
        (index, float(score))
        for index, score in enumerate(scores)
        if score > 0
    ]

    results.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return results


# ---------------------------------------------------------
# MINILM SEMANTIC SEARCH
# ---------------------------------------------------------

def semantic_search(query):
    """Return slides ranked by MiniLM cosine similarity."""

    if not query.strip():
        return []

    query_vector = model.encode(
        query,
        normalize_embeddings=True
    )

    query_vector = np.asarray(query_vector)

    # Stored embeddings and query vector are normalized.
    # Therefore, dot product equals cosine similarity.
    scores = embeddings @ query_vector

    ranked_indices = np.argsort(scores)[::-1]

    return [
        (int(index), float(scores[index]))
        for index in ranked_indices
    ]


# ---------------------------------------------------------
# RECIPROCAL RANK FUSION
# ---------------------------------------------------------

def reciprocal_rank_fusion(retrieval_methods):
    """
    Combine ranked results using Reciprocal Rank Fusion.

    RRF score:
        weight / (RRF_K + rank)

    Raw BM25 and cosine similarity scores are not added.
    Only ranking positions are used.
    """

    combined_scores = {}

    for weight, results in retrieval_methods:

        for rank, (index, _) in enumerate(results, start=1):

            rrf_score = weight / (RRF_K + rank)

            combined_scores[index] = (
                combined_scores.get(index, 0.0)
                + rrf_score
            )

    ranked_results = sorted(
        combined_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return ranked_results


# ---------------------------------------------------------
# SELECTED HYBRID SEARCH: BM25 + MINILM
# ---------------------------------------------------------

def hybrid_search(query, top_k=DEFAULT_TOP_K):
    """
    Selected retrieval method:
        BM25 + MiniLM + RRF

    Weights:
        BM25     = 1
        MiniLM   = 1
        RRF_K    = 60

    Returns:
        [(slide_index, rrf_score), ...]
    """

    if not isinstance(query, str) or not query.strip():
        return []

    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    # Stage 1: Retrieve using BM25.
    bm25_results = bm25_search(query)

    # Stage 2: Retrieve using MiniLM.
    semantic_results = semantic_search(query)

    # Stage 3: Combine the rankings using RRF.
    fused_results = reciprocal_rank_fusion([
        (BM25_WEIGHT, bm25_results),
        (SEMANTIC_WEIGHT, semantic_results),
    ])

    return fused_results[:top_k]


# ---------------------------------------------------------
# RETRIEVAL CONFIGURATION INFORMATION
# ---------------------------------------------------------

def get_retrieval_info():
    """Return the configuration of the selected retrieval method."""

    return {
        "method": "BM25 + MiniLM + Reciprocal Rank Fusion",
        "bm25_weight": BM25_WEIGHT,
        "semantic_weight": SEMANTIC_WEIGHT,
        "rrf_k": RRF_K,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "number_of_slides": len(slides),
        "embedding_shape": list(embeddings.shape),
    }


# ---------------------------------------------------------
# INTERACTIVE TESTING
# ---------------------------------------------------------

def display_results(results):

    for rank, (index, score) in enumerate(results, start=1):

        slide = slides[index]

        print("\n" + "=" * 75)
        print(f"Rank: {rank}")
        print(f"RRF Score: {score:.6f}")
        print("Presentation:", slide.get("presentation_title", ""))
        print("Source:", slide.get("source_file", ""))
        print("Slide number:", slide.get("slide_number", ""))
        print("Slide title:", slide.get("slide_title", ""))
        print("Content:", slide.get("search_text", "")[:400])


def main():

    print("\nBM25 + MINILM HYBRID SEARCH")
    print("Method: Reciprocal Rank Fusion")
    print(f"BM25 weight: {BM25_WEIGHT}")
    print(f"MiniLM weight: {SEMANTIC_WEIGHT}")
    print(f"RRF K: {RRF_K}")
    print("\nType 'exit' to stop.")

    while True:

        query = input("\nEnter your query: ").strip()

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            continue

        results = hybrid_search(query, top_k=DEFAULT_TOP_K)

        display_results(results)


if __name__ == "__main__":
    main()