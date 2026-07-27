import json
from dataclasses import asdict, dataclass
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
        "recall_at_10": evaluation.recall_at_10,
        "reciprocal_rank": evaluation.reciprocal_rank,
    }


@dataclass(frozen=True, slots=True)
class BM25ExperimentSummary:
    experiment: str
    k1: float
    b: float
    split: str
    top_k: int
    query_count: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float


def build_bm25_experiment_summary(
    *,
    experiment: str,
    k1: float,
    b: float,
    split: str,
    top_k: int,
    result: RetrievalEvaluationResult,
) -> BM25ExperimentSummary:
    return BM25ExperimentSummary(
        experiment=experiment,
        k1=k1,
        b=b,
        split=split,
        top_k=top_k,
        query_count=result.query_count,
        recall_at_1=result.recall_at_1,
        recall_at_3=result.recall_at_3,
        recall_at_5=result.recall_at_5,
        recall_at_10=result.recall_at_10,
        mrr=result.mrr,
    )


def export_bm25_experiment_summary(
    summary: BM25ExperimentSummary,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            asdict(summary),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
