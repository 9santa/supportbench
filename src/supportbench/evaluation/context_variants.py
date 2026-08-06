from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from supportbench.evaluation.rag_evaluator import summarize_rag_results


def summarize_context_variants(
    results: Sequence[Mapping[str, Any]],
    *,
    modes: Sequence[str],
) -> dict[str, Any]:
    successful = [result for result in results if result.get("status") == "success"]

    return {
        "query_count": len(results),
        "successful_query_count": len(successful),
        "status_counts": dict(Counter(str(result.get("status")) for result in results)),
        "modes": {mode: _context_metrics(successful, mode=mode) for mode in modes},
        "transitions": {
            f"{left}_to_{right}": _context_transition(
                successful,
                left=left,
                right=right,
            )
            for left, right in zip(modes, modes[1:], strict=False)
        },
    }


def summarize_generation_variants(
    results: Sequence[Mapping[str, Any]],
    *,
    modes: Sequence[str],
) -> dict[str, Any]:
    return {
        "modes": {
            mode: summarize_rag_results(
                [result for result in results if result.get("mode") == mode]
            )
            for mode in modes
        },
        "paired_transitions": {
            f"{left}_to_{right}": _generation_transition(
                results,
                left=left,
                right=right,
            )
            for left, right in zip(modes, modes[1:], strict=False)
        },
    }


def _context_metrics(
    results: Sequence[Mapping[str, Any]],
    *,
    mode: str,
) -> dict[str, float]:
    contexts = [result[mode] for result in results]
    return {
        "gold_document_in_context_rate": _mean_booleans(
            bool(context["gold_document_in_context"])
            for context in contexts
            if context.get("gold_document_in_context") is not None
        ),
        "reference_answer_in_context_rate": _mean_booleans(
            bool(context["reference_answer_in_context"])
            for context in contexts
            if context.get("reference_answer_in_context") is not None
        ),
        "context_truncated_rate": _mean_booleans(
            bool(context["context_truncated"]) for context in contexts
        ),
        "context_tokens_mean": _mean(
            [float(context["context_token_count"]) for context in contexts]
        ),
        "prompt_tokens_mean": _mean([float(context["prompt_token_count"]) for context in contexts]),
    }


def _context_transition(
    results: Sequence[Mapping[str, Any]],
    *,
    left: str,
    right: str,
) -> dict[str, int]:
    pairs = [
        (
            bool(result[left]["reference_answer_in_context"]),
            bool(result[right]["reference_answer_in_context"]),
        )
        for result in results
        if result[left].get("reference_answer_in_context") is not None
        and result[right].get("reference_answer_in_context") is not None
    ]
    return {
        "evaluable_count": len(pairs),
        "gained_count": sum(not before and after for before, after in pairs),
        "lost_count": sum(before and not after for before, after in pairs),
        "tied_present_count": sum(before and after for before, after in pairs),
        "tied_absent_count": sum(not before and not after for before, after in pairs),
    }


def _generation_transition(
    results: Sequence[Mapping[str, Any]],
    *,
    left: str,
    right: str,
) -> dict[str, Any]:
    by_query: dict[str, dict[str, Mapping[str, Any]]] = {}

    for result in results:
        mode = result.get("mode")
        query_id = result.get("query_id")

        if mode in {left, right} and isinstance(query_id, str):
            by_query.setdefault(query_id, {})[str(mode)] = result

    pairs = [
        (variants[left], variants[right])
        for variants in by_query.values()
        if left in variants and right in variants
    ]
    transitions = Counter(f"{_outcome(before)}_to_{_outcome(after)}" for before, after in pairs)
    f1_deltas = [_strict_f1(after) - _strict_f1(before) for before, after in pairs]

    return {
        "paired_query_count": len(pairs),
        "outcome_counts": dict(transitions),
        "answer_gained_count": sum(
            _outcome(before) != "answer" and _outcome(after) == "answer" for before, after in pairs
        ),
        "answer_lost_count": sum(
            _outcome(before) == "answer" and _outcome(after) != "answer" for before, after in pairs
        ),
        "strict_reference_f1_delta_mean": _mean(f1_deltas),
        "strict_reference_f1_improved_count": sum(delta > 1e-12 for delta in f1_deltas),
        "strict_reference_f1_degraded_count": sum(delta < -1e-12 for delta in f1_deltas),
        "strict_reference_f1_tied_count": sum(abs(delta) <= 1e-12 for delta in f1_deltas),
    }


def _outcome(result: Mapping[str, Any]) -> str:
    if result.get("status") == "success" and result.get("decision") is not None:
        return str(result["decision"])

    return str(result.get("status"))


def _strict_f1(result: Mapping[str, Any]) -> float:
    if result.get("status") != "success" or result.get("decision") != "answer":
        return 0.0

    value = result.get("reference_token_f1")
    return float(value) if value is not None else 0.0


def _mean_booleans(values: Iterable[bool]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
