from supportbench.evaluation.rag_evaluator import summarize_rag_results


def _result(
    *,
    status: str,
    reference_status: str,
    decision: str | None,
) -> dict[str, object]:
    return {
        "status": status,
        "benchmark_reference_status": reference_status,
        "decision": decision,
        "llm_called": True,
        "reference_answer": "reference" if reference_status == "answerable" else None,
        "reference_token_precision": 1.0 if decision == "answer" else None,
        "reference_token_recall": 1.0 if decision == "answer" else None,
        "reference_token_f1": 1.0 if decision == "answer" else None,
        "gold_document_in_context": True if reference_status == "answerable" else None,
        "reference_answer_in_context": True if reference_status == "answerable" else None,
        "gold_document_cited": True if decision == "answer" else None,
        "citation_ids": ["parent_a"] if decision == "answer" else [],
        "context_truncated": False,
        "context_token_count": 100,
        "prompt_token_count": 150,
        "context_latency_ms": 10.0,
        "generation_latency_ms": 20.0,
        "total_latency_ms": 30.0,
    }


def test_reference_missing_group_is_descriptive_not_strictly_unanswerable() -> None:
    summary = summarize_rag_results(
        [
            _result(
                status="success",
                reference_status="answerable",
                decision="answer",
            ),
            _result(
                status="success",
                reference_status="benchmark_reference_missing",
                decision="answer",
            ),
        ]
    )

    assert summary["benchmark_reference_missing_query_count"] == 1
    assert summary["decisions"]["benchmark_reference_missing_answer_rate"] == 1.0
    assert "strict_decision_accuracy" not in summary["decisions"]
    assert "abstention_precision" not in summary["decisions"]
    assert "unanswerable_false_answer_rate" not in summary["decisions"]


def test_generation_truncation_is_a_received_but_invalid_response() -> None:
    summary = summarize_rag_results(
        [
            _result(
                status="generation_truncated",
                reference_status="answerable",
                decision=None,
            )
        ]
    )

    assert summary["pipeline"]["llm_response_rate"] == 1.0
    assert summary["pipeline"]["schema_valid_rate"] == 0.0
    assert summary["pipeline"]["generation_truncated_rate"] == 1.0
