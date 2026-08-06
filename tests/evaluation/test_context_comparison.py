from supportbench.evaluation.context_comparison import summarize_context_comparison


def _result(
    *,
    baseline_reference: bool,
    candidate_reference: bool,
    baseline_gold: bool,
    candidate_gold: bool,
) -> dict[str, object]:
    return {
        "status": "success",
        "baseline": {
            "reference_answer_in_context": baseline_reference,
            "gold_document_in_context": baseline_gold,
            "selected_chunk_ids": ["baseline"],
            "context_chunk_ids": ["baseline"],
            "context_truncated": False,
            "context_token_count": 100,
        },
        "candidate": {
            "reference_answer_in_context": candidate_reference,
            "gold_document_in_context": candidate_gold,
            "selected_chunk_ids": ["candidate"],
            "context_chunk_ids": ["candidate"],
            "context_truncated": False,
            "context_token_count": 90,
        },
        "retrieval_and_baseline_latency_ms": 10.0,
        "candidate_latency_ms": 2.0,
    }


def test_summarizes_paired_context_gains_and_losses() -> None:
    summary = summarize_context_comparison(
        [
            _result(
                baseline_reference=False,
                candidate_reference=True,
                baseline_gold=True,
                candidate_gold=True,
            ),
            _result(
                baseline_reference=True,
                candidate_reference=False,
                baseline_gold=True,
                candidate_gold=False,
            ),
            _result(
                baseline_reference=False,
                candidate_reference=False,
                baseline_gold=False,
                candidate_gold=False,
            ),
            _result(
                baseline_reference=True,
                candidate_reference=True,
                baseline_gold=True,
                candidate_gold=True,
            ),
        ]
    )

    comparison = summary["comparison"]

    assert summary["baseline"]["reference_answer_in_context_rate"] == 0.5
    assert summary["candidate"]["reference_answer_in_context_rate"] == 0.5
    assert comparison["reference_gained_count"] == 1
    assert comparison["reference_lost_count"] == 1
    assert comparison["reference_net_gain_count"] == 0
    assert comparison["reference_tied_present_count"] == 1
    assert comparison["reference_tied_absent_count"] == 1
    assert comparison["gold_gained_count"] == 0
    assert comparison["gold_lost_count"] == 1
    assert comparison["selected_chunks_changed_rate"] == 1.0
