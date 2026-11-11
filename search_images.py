
import json
import re

import numpy as np
from rank_bm25 import BM25Okapi

from config import EMBEDDING_DIM, OUTPUT_DIR
from embedder import MiniLMEmbedder


# ---------- Configuration ----------

IMAGE_RECORDS_FILE = OUTPUT_DIR / "image_records.json"
IMAGE_EMBEDDINGS_FILE = OUTPUT_DIR / "image_embeddings.npy"

TOP_K = 10
RRF_K = 60

# RRF weights: these are relative ranking weights, not raw scores.
KEYWORD_WEIGHT = 2
BM25_WEIGHT = 1
SEMANTIC_WEIGHT = 1

# Title words receive more weight than body/context words.
TITLE_KEYWORD_MULTIPLIER = 4

# Small bonuses added to the RRF score.
EXACT_TITLE_BOOST = 0.012
EXACT_CONTEXT_BOOST = 0.006

# Avoid returning too many pictures from one slide.
MAX_RESULTS_PER_SLIDE = 2


# ---------- Load records and embeddings ----------

def load_data():
    if not IMAGE_RECORDS_FILE.exists():
        raise FileNotFoundError(
            f"{IMAGE_RECORDS_FILE} not found. "
            "Run python build_image_records.py first."
        )

    if not IMAGE_EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"{IMAGE_EMBEDDINGS_FILE} not found. "
            "Run python build_image_embeddings.py first."
        )

    with IMAGE_RECORDS_FILE.open("r", encoding="utf-8") as file:
        records = json.load(file)

    embeddings = np.load(IMAGE_EMBEDDINGS_FILE)

    if not records:
        raise ValueError("No image records found.")

    if embeddings.shape != (len(records), EMBEDDING_DIM):
        raise ValueError(
            f"Mismatch: {len(records)} image records, "
            f"embedding shape {embeddings.shape}. "
            "Rebuild image embeddings."
        )

    return records, embeddings


records, embeddings = load_data()

print(f"Loaded {len(records)} image occurrences.")
print(f"Embedding shape: {embeddings.shape}")


# ---------- Text processing ----------

def tokenize_text(text):
    return re.findall(r"\b[a-z0-9]+\b", str(text).lower())


def tokenize(text):
    return set(tokenize_text(text))


def normalize_text(text):
    return " ".join(tokenize_text(text))


def extract_context_body(context):
    """
    Extract slide content and OCR text without the metadata lines.
    This keeps presentation/title metadata from being treated as
    ordinary body text in the keyword search.
    """
    parts = []

    for line in str(context or "").splitlines():
        stripped = line.strip()

        if stripped.startswith("Presentation:"):
            continue

        if stripped.startswith("Slide title:"):
            continue

        if stripped.startswith("Slide content:"):
            value = stripped[len("Slide content:"):].strip()
            if value:
                parts.append(value)
            continue

        if stripped.startswith("Text detected in images:"):
            value = stripped[len("Text detected in images:"):].strip()
            if value:
                parts.append(value)
            continue

        if stripped:
            parts.append(stripped)

    return "\n".join(parts)


titles = [
    str(record.get("slide_title", "") or "")
    for record in records
]

contexts = [
    str(record.get("context_text", "") or "")
    for record in records
]

context_bodies = [
    extract_context_body(context)
    for context in contexts
]

# Use all available searchable text for BM25.
searchable_texts = [
    "\n".join([
        str(record.get("source_file", "") or ""),
        titles[index],
        contexts[index],
    ])
    for index, record in enumerate(records)
]

bm25 = BM25Okapi([
    tokenize_text(text)
    for text in searchable_texts
])

# Load the embedding model once.
model = MiniLMEmbedder()


# ---------- 1. Keyword search ----------

def keyword_search(query):
    query_words = tokenize(query)

    if not query_words:
        return []

    results = []

    for index, record in enumerate(records):
        title_words = tokenize(titles[index])
        body_words = tokenize(context_bodies[index])
        filename_words = tokenize(record.get("source_file", ""))

        title_matches = len(query_words & title_words)
        body_matches = len(query_words & body_words)
        filename_matches = len(query_words & filename_words)

        score = (
            TITLE_KEYWORD_MULTIPLIER * title_matches
            + body_matches
            + filename_matches
        )

        if score > 0:
            results.append((index, float(score)))

    results.sort(key=lambda item: item[1], reverse=True)
    return results


# ---------- 2. BM25 search ----------

def bm25_search(query):
    scores = bm25.get_scores(tokenize_text(query))

    results = [
        (index, float(score))
        for index, score in enumerate(scores)
        if score > 0
    ]

    results.sort(key=lambda item: item[1], reverse=True)
    return results


# ---------- 3. MiniLM semantic search ----------

def semantic_search(query):
    query_vector = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    # Both query and stored embeddings are normalized.
    # Their dot product therefore equals cosine similarity.
    scores = embeddings @ query_vector

    ranked_indices = np.argsort(scores)[::-1]

    return [
        (int(index), float(scores[index]))
        for index in ranked_indices
    ]


# ---------- 4. Exact phrase matching ----------

def exact_match_boost(query, index):
    phrase = normalize_text(query)

    if not phrase:
        return 0.0, False, False

    title = normalize_text(titles[index])
    context = normalize_text(contexts[index])

    # Match a complete normalized phrase, not just individual words.
    title_match = phrase in title
    context_match = phrase in context

    boost = 0.0

    # Give the title priority. Do not stack both bonuses.
    if title_match:
        boost = EXACT_TITLE_BOOST
    elif context_match:
        boost = EXACT_CONTEXT_BOOST

    return boost, title_match, context_match


# ---------- 5. Reciprocal Rank Fusion ----------

def image_search_detailed(query, top_k=TOP_K):
    keyword_results = keyword_search(query)
    bm25_results = bm25_search(query)
    semantic_results = semantic_search(query)

    rankings = [
        ("keyword", keyword_results, KEYWORD_WEIGHT),
        ("bm25", bm25_results, BM25_WEIGHT),
        ("semantic", semantic_results, SEMANTIC_WEIGHT),
    ]

    details = {}

    for method, results, weight in rankings:
        for rank, (index, raw_score) in enumerate(
            results, start=1
        ):
            if index not in details:
                details[index] = {
                    "index": index,
                    "rrf_score": 0.0,
                    "keyword_score": 0.0,
                    "bm25_score": 0.0,
                    "semantic_score": 0.0,
                    "keyword_rank": None,
                    "bm25_rank": None,
                    "semantic_rank": None,
                }

            item = details[index]

            # RRF uses ranks, not the raw score magnitudes.
            item["rrf_score"] += weight / (RRF_K + rank)
            item[f"{method}_score"] = raw_score
            item[f"{method}_rank"] = rank

    for index, item in details.items():
        boost, title_match, context_match = exact_match_boost(
            query, index
        )

        item["exact_title_match"] = title_match
        item["exact_context_match"] = context_match
        item["exact_boost"] = boost
        item["final_score"] = item["rrf_score"] + boost

    ranked = sorted(
        details.values(),
        key=lambda item: (
            item["exact_title_match"],
            item["final_score"],
        ),
        reverse=True,
    )

    # Diversify results so one slide does not dominate.
    diversified = []
    slide_counts = {}

    for item in ranked:
        record = records[item["index"]]

        slide_key = (
            record.get("source_file", ""),
            record.get("slide_number", ""),
        )

        if slide_counts.get(slide_key, 0) >= MAX_RESULTS_PER_SLIDE:
            continue

        diversified.append(item)
        slide_counts[slide_key] = slide_counts.get(slide_key, 0) + 1

        if len(diversified) >= top_k:
            break

    return diversified


# ---------- 6. Display results ----------

def display_results(query, results):
    print(f"\nQuery: {query}")
    print(f"Returned image results: {len(results)}\n")

    if not results:
        print("No matching images found.")
        return

    for rank, item in enumerate(results, start=1):
        record = records[item["index"]]
        image_path = OUTPUT_DIR / record["image_path"]

        print("=" * 65)
        print(f"Rank: {rank}")
        print(f"Final score: {item['final_score']:.6f}")
        print(f"RRF score: {item['rrf_score']:.6f}")
        print(f"Keyword score: {item['keyword_score']:.4f}")
        print(f"BM25 score: {item['bm25_score']:.4f}")
        print(
            "Semantic cosine similarity: "
            f"{item['semantic_score']:.4f}"
        )
        print(f"Exact title match: {item['exact_title_match']}")
        print(f"Exact context match: {item['exact_context_match']}")
        print(f"Exact-match boost: {item['exact_boost']:.4f}")

        print(
            "Method ranks: "
            f"keyword={item['keyword_rank']}, "
            f"BM25={item['bm25_rank']}, "
            f"semantic={item['semantic_rank']}"
        )

        print(f"Image ID: {record.get('image_id', '')}")
        print(f"Presentation: {record.get('source_file', '')}")
        print(f"Slide: {record.get('slide_number', '')}")
        print(f"Slide title: {record.get('slide_title', '')}")
        print(f"Image hash: {record.get('image_hash', '')}")
        print(
            f"Image dimensions: {record.get('width')} "
            f"x {record.get('height')}"
        )
        print(f"Image exists: {image_path.is_file()}")
        print(f"Image path: {image_path}")
        print(f"Context: {contexts[item['index']][:400]}")
        print()


# ---------- Interactive mode ----------

def main():
    print("\nImage hybrid search ready. Type 'exit' to stop.")

    while True:
        query = input("\nSearch for an image: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        results = image_search_detailed(query)
        display_results(query, results)


if __name__ == "__main__":
    main()
