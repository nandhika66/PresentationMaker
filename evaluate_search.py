# OPTIONAL - Evaluates all retrieval methods (Precision@5, Recall@5, MRR)
# using the labelled queries in data/evaluation_queries.json
import json
import re

import numpy as np
from rank_bm25 import BM25Okapi


# ---------- Settings ----------

from config import (
    EMBEDDINGS_FILE,
    EVALUATION_FILE,
    OUTPUT_DIR,
    RECORDS_FILE,
)

TOP_K = 5
RRF_K = 60

BM25_WEIGHT = 1
SEMANTIC_WEIGHT = 1


from embedder import MiniLMEmbedder  # noqa: E402


# ---------- Load data ----------

with open(RECORDS_FILE, "r", encoding="utf-8") as f:
    slides = json.load(f)

with open(EVALUATION_FILE, "r", encoding="utf-8") as f:
    evaluation_queries = json.load(f)

embeddings = np.load(EMBEDDINGS_FILE)

if len(slides) != embeddings.shape[0]:
    raise ValueError(
        f"Slide records ({len(slides)}) and embeddings "
        f"({embeddings.shape[0]}) do not match."
    )

print(f"Loaded {len(slides)} slides")
print(f"Loaded {len(evaluation_queries)} evaluation queries")
print(f"Embeddings shape: {embeddings.shape}")
print(f"Evaluation file: {EVALUATION_FILE}")


# ---------- Load MiniLM ----------

model = MiniLMEmbedder()


# ---------- Tokenization ----------

def tokenize_text(text):
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def tokenize(text):
    return set(tokenize_text(text))


# ---------- Prepare BM25 index ----------

slide_tokens = []

for slide in slides:
    text = " ".join([
        slide.get("presentation_title", ""),
        slide.get("slide_title", ""),
        slide.get("search_text", "")
    ])
    slide_tokens.append(tokenize_text(text))

bm25 = BM25Okapi(slide_tokens)


# ---------- Custom keyword search ----------

def keyword_search(query):
    query_words = tokenize(query)
    results = []

    for index, slide in enumerate(slides):
        title_words = tokenize(
            slide.get("slide_title", "") + " "
            + slide.get("presentation_title", "")
        )

        content_words = tokenize(
            slide.get("search_text", "") + " "
            + slide.get("presentation_title", "")
        )

        score = (
            3 * len(query_words & title_words)
            + len(query_words & content_words)
        )

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


# ---------- MiniLM semantic search ----------

def semantic_search(query):
    query_vector = model.encode(
        query,
        normalize_embeddings=True
    )

    scores = embeddings @ query_vector
    ranked_indices = np.argsort(scores)[::-1]

    return [
        (int(index), float(scores[index]))
        for index in ranked_indices
    ]


# ---------- Hybrid search: BM25 + MiniLM using RRF ----------

def hybrid_search(query):
    bm25_results = bm25_search(query)
    semantic_results = semantic_search(query)

    hybrid_scores = {}

    for rank, (index, _) in enumerate(bm25_results, start=1):
        hybrid_scores[index] = (
            hybrid_scores.get(index, 0)
            + BM25_WEIGHT / (RRF_K + rank)
        )

    for rank, (index, _) in enumerate(
        semantic_results, start=1
    ):
        hybrid_scores[index] = (
            hybrid_scores.get(index, 0)
            + SEMANTIC_WEIGHT / (RRF_K + rank)
        )

    return sorted(
        hybrid_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )


# ---------- Evaluation metrics ----------

def slide_key(slide):
    return (
        slide.get("source_file", "").replace("\\", "/").casefold(),
        int(slide.get("slide_number", 0))
    )


def evaluate_ranking(ranked_indices, relevant_slides):
    relevant_set = {
        (
            item["source_file"].replace("\\", "/").casefold(),
            int(item["slide_number"])
        )
        for item in relevant_slides
    }

    top_indices = ranked_indices[:TOP_K]

    top_keys = [
        slide_key(slides[index])
        for index in top_indices
    ]

    hits = sum(key in relevant_set for key in top_keys)

    # Precision@K uses K as the denominator
    precision = hits / TOP_K

    # Recall is based on all labelled relevant slides
    recall = (
        sum(key in relevant_set for key in set(top_keys))
        / len(relevant_set)
        if relevant_set else 0.0
    )

    # Reciprocal rank of the first relevant result
    reciprocal_rank = 0.0

    for rank, key in enumerate(top_keys, start=1):
        if key in relevant_set:
            reciprocal_rank = 1.0 / rank
            break

    return precision, recall, reciprocal_rank


# ---------- Run evaluation ----------

methods = {
    "Custom Keyword": keyword_search,
    "BM25": bm25_search,
    "MiniLM": semantic_search,
    "Hybrid (BM25 + MiniLM)": hybrid_search,
}

totals = {
    name: {"precision": 0.0, "recall": 0.0, "mrr": 0.0}
    for name in methods
}

details = []

for item in evaluation_queries:
    query = item["query"]
    relevant_slides = item["relevant_slides"]

    # Cache rankings so each method runs once per query
    rankings = {
        name: search_function(query)
        for name, search_function in methods.items()
    }

    query_result = {"query": query, "methods": {}}

    for name, results in rankings.items():
        ranked_indices = [index for index, _ in results]

        precision, recall, reciprocal_rank = evaluate_ranking(
            ranked_indices, relevant_slides
        )

        totals[name]["precision"] += precision
        totals[name]["recall"] += recall
        totals[name]["mrr"] += reciprocal_rank

        query_result["methods"][name] = {
            "precision_at_5": precision,
            "recall_at_5": recall,
            "reciprocal_rank": reciprocal_rank,
            "top_5": [
                {
                    "source_file": slides[index].get("source_file", ""),
                    "slide_number": slides[index].get("slide_number", ""),
                    "slide_title": slides[index].get("slide_title", "")
                }
                for index in ranked_indices[:TOP_K]
            ]
        }

    details.append(query_result)


# ---------- Print summary ----------

query_count = len(evaluation_queries)

if query_count == 0:
    raise ValueError("The evaluation dataset contains no queries.")

print("\n========== EVALUATION SUMMARY ==========")
print(f"Queries evaluated: {query_count}")
print(f"Metrics calculated at K = {TOP_K}\n")

print(
    f"{'Method':<25} {'Precision@5':>12} "
    f"{'Recall@5':>10} {'MRR':>10}"
)
print("-" * 61)

summary = {}

for name, values in totals.items():
    summary[name] = {
        "precision_at_5": values["precision"] / query_count,
        "recall_at_5": values["recall"] / query_count,
        "mrr": values["mrr"] / query_count
    }

    metrics = summary[name]

    print(
        f"{name:<25} "
        f"{metrics['precision_at_5']:>12.3f} "
        f"{metrics['recall_at_5']:>10.3f} "
        f"{metrics['mrr']:>10.3f}"
    )


# ---------- Save detailed results ----------

report = {
    "query_count": query_count,
    "top_k": TOP_K,
    "summary": summary,
    "query_results": details
}

report_file = OUTPUT_DIR / "evaluation_results.json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(report_file, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"\nDetailed results saved to: {report_file}")
