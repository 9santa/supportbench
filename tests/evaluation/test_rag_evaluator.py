from supportbench.evaluation.rag_evaluator import (
    output_contract_diagnostics,
    summarize_rag_results,
)


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
    assert summary["pipeline"]["citation_contract_strict_success_rate"] == 0.5
    assert summary["pipeline"]["citation_contract_valid_rate"] == 0.5
    assert summary["pipeline"]["citation_contract_repaired_count"] == 1
    assert summary["pipeline"]["citation_contract_repaired_rate"] == 0.5
    assert summary["decision_counts"] == {"answer": 1, "abstain": 1}
    assert summary["strict_valid_decision_counts"] == {"answer": 1}


def test_output_contract_diagnostics_detect_answer_violations() -> None:
    answer = " ".join(["word"] * 121) + " Refer to S6. Citation_ids: [S1, S6]"

    diagnostics = output_contract_diagnostics(
        decision="answer",
        answer=answer,
    )

    assert diagnostics["answer_source_id_leak"] is True
    assert diagnostics["answer_embedded_citation_list"] is True
    assert diagnostics["answer_over_120_words"] is True
    assert diagnostics["decision_content_mismatch"] is False


def test_full_output_contract_is_stricter_than_citation_contract() -> None:
    valid = _result(
        status="success",
        reference_status="answerable",
        decision="answer",
    )
    leaking = _result(
        status="success",
        reference_status="answerable",
        decision="answer",
    )
    leaking["parsed_answer"] = "Refer to S1. Citation_ids: [S1]."

    summary = summarize_rag_results([valid, leaking])

    assert summary["pipeline"]["citation_contract_strict_success_rate"] == 1.0
    assert summary["output_contract"]["answer_source_id_leak_rate"] == 0.5
    assert summary["output_contract"]["answer_embedded_citation_list_rate"] == 0.5
    assert summary["output_contract"]["full_output_contract_valid_rate"] == 0.5


def test_decision_content_mismatch_is_a_lexical_diagnostic() -> None:
    detailed_abstain = output_contract_diagnostics(
        decision="abstain",
        answer="Install the fix pack by running setup.exe with the silent option.",
    )
    supported_abstain = output_contract_diagnostics(
        decision="abstain",
        answer="The supplied context does not provide enough information.",
    )
    clarification = output_contract_diagnostics(
        decision="clarify",
        answer="Which product version is installed?",
    )

    assert detailed_abstain["decision_content_mismatch"] is True
    assert supported_abstain["decision_content_mismatch"] is False
    assert clarification["decision_content_mismatch"] is False


def test_context_and_citation_diagnostics_use_conditional_denominators() -> None:
    answered_with_gold_and_citation = _result(
        status="success",
        reference_status="answerable",
        decision="answer",
    )
    answered_without_gold = _result(
        status="success",
        reference_status="answerable",
        decision="answer",
    )
    answered_without_gold["gold_document_in_context"] = False
    answered_without_gold["reference_answer_in_context"] = False
    answered_without_gold["gold_document_cited"] = False
    abstained_with_gold = _result(
        status="success",
        reference_status="answerable",
        decision="abstain",
    )
    answered_with_gold_without_citation = _result(
        status="success",
        reference_status="answerable",
        decision="answer",
    )
    answered_with_gold_without_citation["reference_answer_in_context"] = False
    answered_with_gold_without_citation["gold_document_cited"] = False

    summary = summarize_rag_results(
        [
            answered_with_gold_and_citation,
            answered_without_gold,
            abstained_with_gold,
            answered_with_gold_without_citation,
        ]
    )

    assert summary["context"]["answer_without_gold_context_rate"] == 1 / 3
    assert summary["context"]["abstain_with_gold_context_rate"] == 1 / 3
    assert summary["context"]["abstain_with_reference_in_context_rate"] == 1 / 2
    assert summary["citations"]["gold_citation_hit_given_gold_in_context"] == 1 / 2
