# evidence_selection.py
# Select relevant and diverse source slides using MMR.

import numpy as np

from hybrid_search import embeddings


# Controls relevance versus diversity.
MMR_LAMBDA = 0.7


def normalize_relevance(candidates):
    """Normalize RRF scores to a 0–1 relevance range."""
    scores = np.array(
        [candidate["retrieval_score"] for candidate in candidates],
        dtype=float,
    )

    min_score = scores.min()
    max_score = scores.max()

    if max_score == min_score:
        return np.ones(len(candidates))

    return (scores - min_score) / (max_score - min_score)


def select_evidence(
    candidates,
    top_k,
    lambda_value=MMR_LAMBDA,
):
    """
    Select evidence using Maximal Marginal Relevance.

    Relevance: normalized hybrid retrieval score.
    Diversity: cosine similarity between slide embeddings.

    Returns the selected original slide records.
    """
    if not candidates:
        return []

    if top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    if not 0 <= lambda_value <= 1:
        raise ValueError("lambda_value must be between 0 and 1.")

    top_k = min(top_k, len(candidates))

    # Retrieve the embedding corresponding to each candidate.
    candidate_vectors = np.array([
        embeddings[candidate["_slide_index"]]
        for candidate in candidates
    ])

    relevance_scores = normalize_relevance(candidates)

    selected_indices = []
    remaining_indices = list(range(len(candidates)))

    while remaining_indices and len(selected_indices) < top_k:

        best_index = None
        best_mmr_score = float("-inf")

        for candidate_index in remaining_indices:

            relevance = relevance_scores[candidate_index]

            if not selected_indices:
                diversity_penalty = 0.0
            else:
                similarities = (
                    candidate_vectors[selected_indices]
                    @ candidate_vectors[candidate_index]
                )

                diversity_penalty = float(np.max(similarities))

            mmr_score = (
                lambda_value * relevance
                - (1 - lambda_value) * diversity_penalty
            )

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_index = candidate_index

        selected_indices.append(best_index)
        remaining_indices.remove(best_index)

    return [
        candidates[index]
        for index in selected_indices
    ]