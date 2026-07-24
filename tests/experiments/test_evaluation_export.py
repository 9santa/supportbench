import json
from pathlib import Path
from typing import Any

from supportbench.evaluation.evaluation_export import export_query_evaluations
from supportbench.evaluation.retrieval_evaluator import (
    QueryEvaluation,
    RetrievalEvaluationResult,
)


def test_exports_query_evaluations_as_jsonl(
    tmp_path: Path,
) -> None:
    result = RetrievalEvaluationResult(
        query_count=1,
        recall_at_1=1.0,
        recall_at_3=1.0,
        recall_at_5=1.0,
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
                recall_at_1=1.0,
                recall_at_3=1.0,
                recall_at_5=1.0,
                reciprocal_rank=1.0,
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
            "gold_rank": 1,
            "recall_at_1": 1.0,
            "recall_at_3": 1.0,
            "recall_at_5": 1.0,
            "reciprocal_rank": 1.0,
        }
    ]


def test_export_creates_parent_directories(
    tmp_path: Path,
) -> None:
    result = RetrievalEvaluationResult(
        query_count=0,
        recall_at_1=0.0,
        recall_at_3=0.0,
        recall_at_5=0.0,
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
