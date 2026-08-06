from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from math import ceil, floor
from typing import Any


def summarize_context_comparison(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    successful = [result for result in results if result.get("status") == "success"]
    status_counts = Counter(str(result.get("status")) for result in results)

    reference_pairs = _boolean_pairs(
        successful,
        key="reference_answer_in_context",
    )
    gold_pairs = _boolean_pairs(
        successful,
        key="gold_document_in_context",
    )

    return {
        "query_count": len(results),
        "successful_query_count": len(successful),
        "status_counts": dict(status_counts),
        "baseline": _context_metrics(successful, side="baseline"),
        "candidate": _context_metrics(successful, side="candidate"),
        "comparison": {
            **_transition_metrics(reference_pairs, prefix="reference"),
            **_transition_metrics(gold_pairs, prefix="gold"),
            "selected_chunks_changed_count": sum(
                result["baseline"]["selected_chunk_ids"]
                != result["candidate"]["selected_chunk_ids"]
                for result in successful
            ),
            "selected_chunks_changed_rate": _mean_booleans(
                result["baseline"]["selected_chunk_ids"]
                != result["candidate"]["selected_chunk_ids"]
                for result in successful
            ),
            "rendered_chunks_changed_count": sum(
                result["baseline"]["context_chunk_ids"]
                != result["candidate"]["context_chunk_ids"]
                for result in successful
            ),
            "rendered_chunks_changed_rate": _mean_booleans(
                result["baseline"]["context_chunk_ids"]
                != result["candidate"]["context_chunk_ids"]
                for result in successful
            ),
        },
        "latency": {
            **_distribution(
                [
                    float(result["retrieval_and_baseline_latency_ms"])
                    for result in successful
                ],
                prefix="retrieval_and_baseline_ms",
            ),
            **_distribution(
                [float(result["candidate_latency_ms"]) for result in successful],
                prefix="candidate_ms",
            ),
        },
    }


def _context_metrics(
    results: Sequence[Mapping[str, Any]],
    *,
    side: str,
) -> dict[str, float]:
    contexts = [result[side] for result in results]

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
    }


def _boolean_pairs(
    results: Sequence[Mapping[str, Any]],
    *,
    key: str,
) -> list[tuple[bool, bool]]:
    return [
        (bool(result["baseline"][key]), bool(result["candidate"][key]))
        for result in results
        if result["baseline"].get(key) is not None
        and result["candidate"].get(key) is not None
    ]


def _transition_metrics(
    pairs: Sequence[tuple[bool, bool]],
    *,
    prefix: str,
) -> dict[str, int | float]:
    gained = sum(not baseline and candidate for baseline, candidate in pairs)
    lost = sum(baseline and not candidate for baseline, candidate in pairs)
    tied_present = sum(baseline and candidate for baseline, candidate in pairs)
    tied_absent = sum(not baseline and not candidate for baseline, candidate in pairs)

    return {
        f"{prefix}_evaluable_count": len(pairs),
        f"{prefix}_gained_count": gained,
        f"{prefix}_gained_rate": _safe_divide(gained, len(pairs)),
        f"{prefix}_lost_count": lost,
        f"{prefix}_lost_rate": _safe_divide(lost, len(pairs)),
        f"{prefix}_net_gain_count": gained - lost,
        f"{prefix}_tied_present_count": tied_present,
        f"{prefix}_tied_absent_count": tied_absent,
    }


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _mean_booleans(values: Iterable[bool]) -> float:
    items = list(values)
    return _safe_divide(sum(bool(value) for value in items), len(items))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _distribution(
    values: Sequence[float],
    *,
    prefix: str,
) -> dict[str, float]:
    if not values:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p95": 0.0,
        }

    return {
        f"{prefix}_mean": _mean(values),
        f"{prefix}_p50": _percentile(values, 0.50),
        f"{prefix}_p95": _percentile(values, 0.95),
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = floor(position)
    upper = ceil(position)

    if lower == upper:
        return ordered[lower]

    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
