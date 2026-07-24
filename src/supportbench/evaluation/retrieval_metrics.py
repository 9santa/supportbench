def recall_at_k(
    retrieved_doc_ids: list[str],
    relevant_doc_ids: set[str],
    k: int,
) -> float:
    """Return the fraction of relevant documents retrieved in top-k."""
    if k <= 0:
        raise ValueError("k must be positive")

    if not relevant_doc_ids:
        raise ValueError("relevant_doc_ids must not be empty")

    retrieved_at_k = set(retrieved_doc_ids[:k])
    # set intersection
    relevant_retrieved = retrieved_at_k & relevant_doc_ids

    return len(relevant_retrieved) / len(relevant_doc_ids)


def reciprocal_rank(
    retrieved_doc_ids: list[str],
    relevant_doc_ids: set[str],
) -> float:
    """Return the reciprocal rank of the first relevant document."""
    if not relevant_doc_ids:
        raise ValueError("relevant_doc_ids must not be empty")

    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(ranks: list[float]) -> float:
    """Return the arithmetic mean of reciprocal-ranks."""
    if not ranks:
        return 0.0

    return sum(ranks) / len(ranks)
