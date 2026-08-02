import json
from pathlib import Path
from typing import Any

from supportbench.evaluation.retrieval_evaluator import (
    QueryEvaluation,
    RetrievalEvaluationResult,
)
from supportbench.experiments.evaluation_export import (
    build_bm25_experiment_summary,
    export_bm25_experiment_summary,
    export_query_evaluations,
)


def test_exports_query_evaluations_as_jsonl(
    tmp_path: Path,
) -> None:
    result = RetrievalEvaluationResult(
        query_count=1,
        labeled_query_count=1,
        unlabeled_query_count=0,
        evaluation_top_k=10,
        recall_cutoffs=(1, 3, 5, 10),
        mrr_cutoff=10,
        recalls=(
            (1, 1.0),
            (3, 1.0),
            (5, 1.0),
            (10, 1.0),
        ),
        mrr=1.0,
        queries=(
            QueryEvaluation(
                query_id="q0001",
                query="как настроить vpn",
                relevant_doc_ids=("vpn_setup",),
                retrieved_doc_ids=(
                    "vpn_setup",
                    "vpn_faq",
                ),
                scores=(4.82, 3.17),
                is_labeled=True,
                recalls=(
                    (1, 1.0),
                    (3, 1.0),
                    (5, 1.0),
                    (10, 1.0),
                ),
                reciprocal_rank=1.0,
                mrr_cutoff=10,
            ),
        ),
    )

    output_path = tmp_path / "queries.jsonl"

    export_query_evaluations(result, output_path)

    records: list[dict[str, Any]] = [
        json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert records == [
        {
            "query_id": "q0001",
            "query": "как настроить vpn",
            "relevant_doc_ids": ["vpn_setup"],
            "retrieved_doc_ids": [
                "vpn_setup",
                "vpn_faq",
            ],
            "scores": [4.82, 3.17],
            "is_labeled": True,
            "gold_rank": 1,
            "recall_at_1": 1.0,
            "recall_at_3": 1.0,
            "recall_at_5": 1.0,
            "recall_at_10": 1.0,
            "reciprocal_rank": 1.0,
            "mrr_cutoff": 10,
        }
    ]


def test_export_creates_parent_directories(
    tmp_path: Path,
) -> None:
    result = RetrievalEvaluationResult(
        query_count=0,
        labeled_query_count=0,
        unlabeled_query_count=0,
        evaluation_top_k=10,
        recall_cutoffs=(1, 3, 5, 10),
        mrr_cutoff=10,
        recalls=(
            (1, 0.0),
            (3, 0.0),
            (5, 0.0),
            (10, 0.0),
        ),
        mrr=0.0,
        queries=(),
    )
    output_path = tmp_path / "bm25" / "b_0_75" / "queries.jsonl"

    export_query_evaluations(
        result,
        output_path,
    )

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == ""


def test_exports_bm25_experiment_summary(
    tmp_path: Path,
) -> None:
    result = RetrievalEvaluationResult(
        query_count=2,
        labeled_query_count=2,
        unlabeled_query_count=0,
        evaluation_top_k=50,
        recall_cutoffs=(1, 3, 5, 10, 20, 50),
        mrr_cutoff=10,
        recalls=(
            (1, 0.5),
            (3, 0.75),
            (5, 1.0),
            (10, 1.0),
            (20, 1.0),
            (50, 1.0),
        ),
        mrr=0.625,
        queries=(),
    )

    summary = build_bm25_experiment_summary(
        experiment="b_0_50",
        k1=1.5,
        b=0.5,
        split="dev",
        top_k=50,
        result=result,
    )

    output_path = tmp_path / "summary.json"

    export_bm25_experiment_summary(
        summary,
        output_path,
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "experiment": "b_0_50",
        "k1": 1.5,
        "b": 0.5,
        "split": "dev",
        "top_k": 50,
        "query_count": 2,
        "labeled_query_count": 2,
        "unlabeled_query_count": 0,
        "recall_at_1": 0.5,
        "recall_at_3": 0.75,
        "recall_at_5": 1.0,
        "recall_at_10": 1.0,
        "recall_at_20": 1.0,
        "recall_at_50": 1.0,
        "mrr": 0.625,
    }
