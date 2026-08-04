from supportbench.evaluation.retrieval_evaluator import (
    RetrievalEvaluationResult,
)


def retrieval_metrics(
    result: RetrievalEvaluationResult,
    *,
    prefix: str,
) -> dict[str, float]:
    metrics = {f"{prefix}_hit_at_{cutoff}": value for cutoff, value in result.recalls}

    metrics[f"{prefix}_mrr_at_{result.mrr_cutoff}"] = result.mrr

    return metrics
