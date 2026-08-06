from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

ORACLE_CONTEXT_MODES = (
    "current",
    "gold_injected",
    "gold_only_selected",
    "oracle_source",
)


def summarize_oracle_contexts(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    successful = [result for result in results if result.get("status") == "success"]

    mode_metrics = {
        mode: _mode_metrics(successful, mode=mode)
        for mode in ORACLE_CONTEXT_MODES
    }
    transitions = {}

    for left, right in zip(
        ORACLE_CONTEXT_MODES,
        ORACLE_CONTEXT_MODES[1:],
        strict=False,
    ):
        transitions[f"{left}_to_{right}"] = _transition_metrics(
            _reference_pairs(successful, left=left, right=right)
        )

    current_gold = [
        result
        for result in successful
        if result["current"].get("gold_document_in_context") is True
    ]
    current_gold_without_reference = [
        result
        for result in current_gold
        if result["current"].get("reference_answer_in_context") is False
    ]
    oracle_verifiable = [
        result
        for result in successful
        if result["oracle_source"].get("reference_answer_in_context") is True
    ]

    return {
        "query_count": len(results),
        "successful_query_count": len(successful),
        "status_counts": dict(
            Counter(str(result.get("status")) for result in results)
        ),
        "modes": mode_metrics,
        "transitions": transitions,
        "diagnostics": {
            "current_gold_missing_count": sum(
                result["current"].get("gold_document_in_context") is False
                for result in successful
            ),
            "current_gold_without_reference_count": len(
                current_gold_without_reference
            ),
            "current_gold_without_reference_rate": _safe_divide(
                len(current_gold_without_reference),
                len(current_gold),
            ),
            "oracle_reference_in_full_source_count": sum(
                result["oracle_source"].get("reference_in_full_source") is True
                for result in successful
            ),
            "oracle_reference_in_rendered_context_count": len(oracle_verifiable),
            "oracle_generator_attribution_query_count": len(oracle_verifiable),
        },
    }


def _mode_metrics(
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
            bool(context["context_truncated"])
            for context in contexts
        ),
        "context_tokens_mean": _mean(
            [float(context["context_token_count"]) for context in contexts]
        ),
        "prompt_tokens_mean": _mean(
            [float(context["prompt_token_count"]) for context in contexts]
        ),
    }


def _reference_pairs(
    results: Sequence[Mapping[str, Any]],
    *,
    left: str,
    right: str,
) -> list[tuple[bool, bool]]:
    return [
        (
            bool(result[left]["reference_answer_in_context"]),
            bool(result[right]["reference_answer_in_context"]),
        )
        for result in results
        if result[left].get("reference_answer_in_context") is not None
        and result[right].get("reference_answer_in_context") is not None
    ]


def _transition_metrics(pairs: Sequence[tuple[bool, bool]]) -> dict[str, int | float]:
    gained = sum(not left and right for left, right in pairs)
    lost = sum(left and not right for left, right in pairs)

    return {
        "evaluable_count": len(pairs),
        "gained_count": gained,
        "gained_rate": _safe_divide(gained, len(pairs)),
        "lost_count": lost,
        "lost_rate": _safe_divide(lost, len(pairs)),
        "net_gain_count": gained - lost,
        "tied_present_count": sum(left and right for left, right in pairs),
        "tied_absent_count": sum(not left and not right for left, right in pairs),
    }


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _mean_booleans(values: Iterable[bool]) -> float:
    items = list(values)
    return _safe_divide(sum(items), len(items))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
