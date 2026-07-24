import json
from pathlib import Path
from typing import Any

from supportbench.evaluation.retrieval_evaluator import (
    QueryEvaluation,
    RetrievalEvaluationResult,
)


def export_query_evaluations(
    result: RetrievalEvaluationResult,
    path: Path,
) -> None:
    """Write one query evaluation per JSONL line."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(mode="w", encoding="utf-8") as file:
        for eval in result.queries:
            record = _query_evaluation_to_dict(eval)

            file.write(json.dumps(record, ensure_ascii=False))
            file.write("\n")


def _query_evaluation_to_dict(evaluation: QueryEvaluation) -> dict[str, Any]:
    return {
        "query_id": evaluation.query_id,
        "query": evaluation.query,
        "relevant_doc_ids": list(evaluation.relevant_doc_ids),
        "retrieved_doc_ids": list(evaluation.retrieved_doc_ids),
        "scores": list(evaluation.scores),
        "gold_rank": evaluation.first_relevant_rank,
        "recall_at_1": evaluation.recall_at_1,
        "recall_at_3": evaluation.recall_at_3,
        "recall_at_5": evaluation.recall_at_5,
        "reciprocal_rank": evaluation.reciprocal_rank,
    }
