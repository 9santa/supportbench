from supportbench.evaluation.oracle_context import summarize_oracle_contexts


def _mode(*, gold: bool, reference: bool) -> dict[str, object]:
    return {
        "gold_document_in_context": gold,
        "reference_answer_in_context": reference,
        "context_truncated": False,
        "context_token_count": 100,
        "prompt_token_count": 150,
    }


def test_summarizes_oracle_context_transitions() -> None:
    result = {
        "status": "success",
        "current": _mode(gold=False, reference=False),
        "gold_injected": _mode(gold=True, reference=False),
        "gold_only_selected": _mode(gold=True, reference=False),
        "oracle_source": {
            **_mode(gold=True, reference=True),
            "reference_in_full_source": True,
        },
    }

    summary = summarize_oracle_contexts([result])

    assert summary["modes"]["current"]["gold_document_in_context_rate"] == 0.0
    assert summary["modes"]["gold_injected"]["gold_document_in_context_rate"] == 1.0
    assert (
        summary["transitions"]["gold_only_selected_to_oracle_source"][
            "gained_count"
        ]
        == 1
    )
    assert summary["diagnostics"]["current_gold_missing_count"] == 1
    assert summary["diagnostics"]["oracle_generator_attribution_query_count"] == 1
