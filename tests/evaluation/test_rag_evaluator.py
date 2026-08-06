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
        "parsed_decision": decision,
        "parsed_answer": "generated answer" if decision is not None else None,
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
        "raw_citation_ids": ["S1"] if decision == "answer" else [],
        "resolved_citation_ids": ["parent_a"] if decision == "answer" else [],
        "contract_repaired": False,
        "strict_contract_valid": status == "success",
        "contract_violations": [],
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


def test_generated_and_strict_valid_decisions_are_counted_separately() -> None:
    contract_error = _result(
        status="citation_contract_error",
        reference_status="answerable",
        decision=None,
    )
    contract_error["parsed_decision"] = "abstain"
    contract_error["parsed_answer"] = "Not enough evidence."

    summary = summarize_rag_results(
        [
            _result(
                status="success",
                reference_status="answerable",
                decision="answer",
            ),
            contract_error,
        ]
    )

    assert summary["generated_decision_counts"] == {
        "answer": 1,
        "abstain": 1,
    }
    assert summary["strict_valid_decision_counts"] == {"answer": 1}
    assert summary["pipeline"]["citation_resolution_valid_rate"] == 1.0
    assert summary["pipeline"]["citation_contract_valid_rate"] == 0.5


def test_repaired_contract_is_operational_success_but_not_strict_success() -> None:
    repaired = _result(
        status="success",
        reference_status="benchmark_reference_missing",
        decision="abstain",
    )
    repaired["raw_citation_ids"] = ["S1", "S2"]
    repaired["resolved_citation_ids"] = ["parent_a", "parent_b"]
    repaired["contract_repaired"] = True
    repaired["strict_contract_valid"] = False
    repaired["contract_violations"] = ["non_answer_has_citations"]

    summary = summarize_rag_results(
        [
            _result(
                status="success",
                reference_status="answerable",
                decision="answer",
            ),
            repaired,
        ]
    )

    assert summary["pipeline"]["generation_success_rate"] == 1.0
    assert summary["pipeline"]["strict_generation_success_rate"] == 0.5
    assert summary["pipeline"]["citation_contract_valid_rate"] == 0.5
    assert summary["pipeline"]["citation_contract_repaired_count"] == 1
    assert summary["pipeline"]["citation_contract_repaired_rate"] == 0.5
    assert summary["decision_counts"] == {"answer": 1, "abstain": 1}
    assert summary["strict_valid_decision_counts"] == {"answer": 1}
