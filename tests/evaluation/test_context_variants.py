from supportbench.evaluation.context_variants import (
    summarize_context_variants,
    summarize_generation_variants,
)


def test_summarizes_context_and_generation_variant_transitions() -> None:
    context_results = [
        {
            "status": "success",
            "top_1": _context(gold=False, reference=False, tokens=100),
            "top_2": _context(gold=True, reference=True, tokens=200),
        }
    ]
    generation_results = [
        _generation(mode="top_1", decision="abstain", f1=None),
        _generation(mode="top_2", decision="answer", f1=0.5),
    ]

    context_summary = summarize_context_variants(
        context_results,
        modes=("top_1", "top_2"),
    )
    generation_summary = summarize_generation_variants(
        generation_results,
        modes=("top_1", "top_2"),
    )

    assert context_summary["modes"]["top_2"]["gold_document_in_context_rate"] == 1.0
    assert context_summary["transitions"]["top_1_to_top_2"]["gained_count"] == 1
    transition = generation_summary["paired_transitions"]["top_1_to_top_2"]
    assert transition["answer_gained_count"] == 1
    assert transition["strict_reference_f1_delta_mean"] == 0.5


def _context(*, gold: bool, reference: bool, tokens: int) -> dict[str, object]:
    return {
        "gold_document_in_context": gold,
        "reference_answer_in_context": reference,
        "context_truncated": False,
        "context_token_count": tokens,
        "prompt_token_count": tokens + 20,
    }


def _generation(
    *,
    mode: str,
    decision: str,
    f1: float | None,
) -> dict[str, object]:
    return {
        "query_id": "q1",
        "mode": mode,
        "status": "success",
        "decision": decision,
        "benchmark_reference_status": "answerable",
        "reference_token_f1": f1,
        "strict_contract_valid": True,
        "contract_repaired": False,
        "citation_ids": ["gold"] if decision == "answer" else [],
        "gold_document_in_context": True,
        "reference_answer_in_context": True,
        "gold_document_cited": decision == "answer",
        "context_truncated": False,
        "context_token_count": 100,
        "prompt_token_count": 120,
        "llm_called": True,
        "generation_latency_ms": 1.0,
        "total_latency_ms": 1.0,
    }
