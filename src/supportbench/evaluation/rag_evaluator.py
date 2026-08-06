import html
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from math import ceil, floor, isfinite
from typing import Any

SUCCESS_STATUS = "success"
RESPONSE_STATUSES = {
    "success",
    "generation_truncated",
    "parse_error",
    "citation_error",
    "citation_resolution_error",
    "citation_contract_error",
}
REFERENCE_AVAILABLE = "answerable"
REFERENCE_MISSING = "benchmark_reference_missing"
SOURCE_ID_PATTERN = re.compile(r"\bS\d+\b")
EMBEDDED_CITATION_LIST_PATTERN = re.compile(
    r"\bcitation[\s_-]*ids?\b",
    re.IGNORECASE,
)
ABSTENTION_SIGNAL_PATTERN = re.compile(
    r"\b(?:"
    r"cannot|can't|unable|insufficient|not enough|not available|not provided|not found|"
    r"does not (?:address|contain|provide)|do not (?:have|provide)|"
    r"no (?:direct|relevant|specific|sufficient) (?:answer|context|documents?|evidence|information)"
    r")\b",
    re.IGNORECASE,
)
CLARIFICATION_SIGNAL_PATTERN = re.compile(
    r"\b(?:clarify|could you|please (?:provide|specify)|what|which)\b",
    re.IGNORECASE,
)


def normalize_reference_text(text: str) -> str:
    value = html.unescape(text)
    value = unicodedata.normalize("NFKC", value)
    value = value.casefold()
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def reference_is_in_text(reference_answer: str | None, text: str) -> bool | None:
    if reference_answer is None or not reference_answer.strip():
        return None

    reference = normalize_reference_text(reference_answer)
    normalized_text = normalize_reference_text(text)

    if not reference:
        return None

    return reference in normalized_text


def lexical_token_scores(
    generated_answer: str,
    reference_answer: str | None,
) -> tuple[float | None, float | None, float | None]:
    """Returns (precision, recall, f1) for tokens."""
    if reference_answer is None or not reference_answer.strip():
        return None, None, None

    predicted_tokens = _word_tokens(generated_answer)
    reference_tokens = _word_tokens(reference_answer)

    if not predicted_tokens:
        return 0.0, 0.0, 0.0

    if not reference_tokens:
        return None, None, None

    prediction_counts = Counter(predicted_tokens)
    reference_counts = Counter(reference_tokens)

    overlap = sum((prediction_counts & reference_counts).values())

    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)

    f1 = _f1(precision, recall)

    return precision, recall, f1


def output_contract_diagnostics(
    *,
    decision: str | None,
    answer: str | None,
) -> dict[str, bool | int]:
    text = answer or ""
    word_count = len(_word_tokens(text))
    source_id_leak = bool(SOURCE_ID_PATTERN.search(text))
    embedded_citation_list = bool(EMBEDDED_CITATION_LIST_PATTERN.search(text))
    over_120_words = word_count > 120

    if decision == "abstain":
        decision_content_mismatch = not bool(ABSTENTION_SIGNAL_PATTERN.search(text))
    elif decision == "clarify":
        decision_content_mismatch = not (
            "?" in text or bool(CLARIFICATION_SIGNAL_PATTERN.search(text))
        )
    else:
        decision_content_mismatch = False

    return {
        "answer_source_id_leak": source_id_leak,
        "answer_embedded_citation_list": embedded_citation_list,
        "answer_word_count": word_count,
        "answer_over_120_words": over_120_words,
        "decision_content_mismatch": decision_content_mismatch,
    }


def summarize_rag_results(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    total = len(results)

    status_counts = Counter(str(result["status"]) for result in results)

    successful = [result for result in results if result["status"] == SUCCESS_STATUS]
    strict_successful = [result for result in successful if _strict_contract_valid(result)]
    contract_repaired_count = sum(
        bool(result.get("contract_repaired")) for result in results
    )

    answerable = [
        result for result in results if _reference_status(result) == REFERENCE_AVAILABLE
    ]

    reference_missing = [
        result for result in results if _reference_status(result) == REFERENCE_MISSING
    ]

    decisions = Counter(
        str(result["decision"]) for result in successful if result.get("decision") is not None
    )
    strict_valid_decisions = Counter(
        str(result["decision"])
        for result in strict_successful
        if result.get("decision") is not None
    )
    generated_decisions = Counter(
        decision
        for result in results
        if (decision := _generated_decision(result)) is not None
    )
    generated_outputs = [
        result
        for result in results
        if _generated_decision(result) is not None and _generated_answer(result) is not None
    ]
    generated_answers = [
        result for result in generated_outputs if _generated_decision(result) == "answer"
    ]
    output_diagnostics = {
        id(result): output_contract_diagnostics(
            decision=_generated_decision(result),
            answer=_generated_answer(result),
        )
        for result in generated_outputs
    }
    full_output_contract_valid_count = sum(
        result["status"] == SUCCESS_STATUS
        and _strict_contract_valid(result)
        and not _has_output_contract_violation(output_diagnostics[id(result)])
        for result in generated_outputs
    )

    response_received = sum(result["status"] in RESPONSE_STATUSES for result in results)

    schema_valid = sum(
        result["status"]
        in {
            "success",
            "citation_error",
            "citation_resolution_error",
            "citation_contract_error",
        }
        for result in results
    )
    citations_resolved = sum(
        result["status"]
        in {
            "success",
            "citation_contract_error",
        }
        for result in results
    )

    llm_called = sum(bool(result.get("llm_called")) for result in results)

    answerable_answer = _count(
        answerable,
        decision="answer",
        status="success",
    )
    answerable_abstain = _count(
        answerable,
        decision="abstain",
        status="success",
    )
    answerable_clarify = _count(
        answerable,
        decision="clarify",
        status="success",
    )

    reference_missing_answer = _count(
        reference_missing,
        decision="answer",
        status="success",
    )
    reference_missing_abstain = _count(
        reference_missing,
        decision="abstain",
        status="success",
    )
    reference_missing_clarify = _count(
        reference_missing,
        decision="clarify",
        status="success",
    )

    successful_answers = [result for result in successful if result.get("decision") == "answer"]

    answered_answerable = [
        result
        for result in answerable
        if result["status"] == "success" and result.get("decision") == "answer"
    ]

    reference_labeled = [result for result in answerable if result.get("reference_answer")]

    answered_reference_scores = [
        float(result["reference_token_f1"])
        for result in answered_answerable
        if result.get("reference_token_f1") is not None
    ]

    strict_reference_scores = [
        (
            float(result["reference_token_f1"])
            if (
                result["status"] == "success"
                and result.get("decision") == "answer"
                and result.get("reference_token_f1") is not None
            )
            else 0.0
        )
        for result in reference_labeled
    ]

    gold_context_results = [
        result for result in answerable if result.get("gold_document_in_context") is not None
    ]

    reference_context_results = [
        result for result in answerable if result.get("reference_answer_in_context") is not None
    ]

    gold_citation_results = [
        result for result in answered_answerable if result.get("gold_document_cited") is not None
    ]
    answered_with_gold_context = [
        result for result in answered_answerable if result.get("gold_document_in_context") is True
    ]
    gold_in_context = [
        result for result in answerable if result.get("gold_document_in_context") is True
    ]
    reference_in_context = [
        result for result in answerable if result.get("reference_answer_in_context") is True
    ]

    return {
        "query_count": total,
        "answerable_query_count": len(answerable),
        "benchmark_reference_missing_query_count": len(reference_missing),
        "status_counts": dict(status_counts),
        "decision_counts": dict(decisions),
        "generated_decision_counts": dict(generated_decisions),
        "strict_valid_decision_counts": dict(strict_valid_decisions),
        "pipeline": {
            "generation_success_rate": (
                _safe_divide(
                    len(successful),
                    total,
                )
            ),
            "citation_contract_strict_success_rate": (
                _safe_divide(
                    len(strict_successful),
                    total,
                )
            ),
            "llm_response_rate": (
                _safe_divide(
                    response_received,
                    llm_called,
                )
            ),
            "schema_valid_rate": (
                _safe_divide(
                    schema_valid,
                    response_received,
                )
            ),
            "citation_valid_rate": (
                _safe_divide(
                    len(successful),
                    schema_valid,
                )
            ),
            "citation_resolution_valid_rate": (
                _safe_divide(
                    citations_resolved,
                    schema_valid,
                )
            ),
            "citation_contract_valid_rate": (
                _safe_divide(
                    len(strict_successful),
                    citations_resolved,
                )
            ),
            "citation_contract_repaired_count": contract_repaired_count,
            "citation_contract_repaired_rate": (
                _safe_divide(
                    contract_repaired_count,
                    total,
                )
            ),
            "generation_truncated_rate": (
                _safe_divide(
                    status_counts["generation_truncated"],
                    llm_called,
                )
            ),
        },
        "output_contract": {
            "answer_source_id_leak_rate": _mean_booleans(
                bool(output_diagnostics[id(result)]["answer_source_id_leak"])
                for result in generated_answers
            ),
            "answer_embedded_citation_list_rate": _mean_booleans(
                bool(output_diagnostics[id(result)]["answer_embedded_citation_list"])
                for result in generated_answers
            ),
            "answer_over_120_words_rate": _mean_booleans(
                bool(output_diagnostics[id(result)]["answer_over_120_words"])
                for result in generated_answers
            ),
            "decision_content_mismatch_rate": _mean_booleans(
                bool(output_diagnostics[id(result)]["decision_content_mismatch"])
                for result in generated_outputs
            ),
            "full_output_contract_valid_rate": _safe_divide(
                full_output_contract_valid_count,
                total,
            ),
        },
        "decisions": {
            "answer_rate": _safe_divide(
                decisions["answer"],
                len(successful),
            ),
            "abstention_rate": (
                _safe_divide(
                    decisions["abstain"],
                    len(successful),
                )
            ),
            "clarification_rate": (
                _safe_divide(
                    decisions["clarify"],
                    len(successful),
                )
            ),
            "answerable_answer_rate": (
                _safe_divide(
                    answerable_answer,
                    len(answerable),
                )
            ),
            "answerable_abstention_rate": (
                _safe_divide(
                    answerable_abstain,
                    len(answerable),
                )
            ),
            "answerable_clarification_rate": (
                _safe_divide(
                    answerable_clarify,
                    len(answerable),
                )
            ),
            "benchmark_reference_missing_answer_rate": (
                _safe_divide(
                    reference_missing_answer,
                    len(reference_missing),
                )
            ),
            "benchmark_reference_missing_abstention_rate": (
                _safe_divide(
                    reference_missing_abstain,
                    len(reference_missing),
                )
            ),
            "benchmark_reference_missing_clarification_rate": (
                _safe_divide(
                    reference_missing_clarify,
                    len(reference_missing),
                )
            ),
        },
        "context": {
            "gold_document_in_context_rate": (
                _mean_booleans(
                    result["gold_document_in_context"] for result in gold_context_results
                )
            ),
            "reference_answer_in_context_rate": (
                _mean_booleans(
                    result["reference_answer_in_context"] for result in reference_context_results
                )
            ),
            "answer_without_gold_context_rate": _mean_booleans(
                result.get("gold_document_in_context") is False
                for result in answered_answerable
                if result.get("gold_document_in_context") is not None
            ),
            "abstain_with_gold_context_rate": _mean_booleans(
                result["status"] == SUCCESS_STATUS and result.get("decision") == "abstain"
                for result in gold_in_context
            ),
            "abstain_with_reference_in_context_rate": _mean_booleans(
                result["status"] == SUCCESS_STATUS and result.get("decision") == "abstain"
                for result in reference_in_context
            ),
            "context_truncated_rate": (
                _mean_booleans(
                    bool(result.get("context_truncated"))
                    for result in results
                    if result.get("context_truncated") is not None
                )
            ),
            **_distribution(
                [
                    float(result["context_token_count"])
                    for result in results
                    if result.get("context_token_count") is not None
                ],
                prefix="context_tokens",
            ),
            **_distribution(
                [
                    float(result["prompt_token_count"])
                    for result in results
                    if result.get("prompt_token_count") is not None
                ],
                prefix="prompt_tokens",
            ),
        },
        "citations": {
            "gold_parent_citation_hit_rate": (
                _mean_booleans(result["gold_document_cited"] for result in gold_citation_results)
            ),
            "gold_citation_hit_given_gold_in_context": _mean_booleans(
                result.get("gold_document_cited") is True
                for result in answered_with_gold_context
            ),
            "mean_citations_per_answer": (
                _mean(
                    [
                        float(
                            len(
                                result.get(
                                    "citation_ids",
                                    (),
                                )
                            )
                        )
                        for result in successful_answers
                    ]
                )
            ),
        },
        "reference_lexical": {
            "reference_token_f1_mean_answered": (_mean(answered_reference_scores)),
            "reference_token_f1_mean_strict": (_mean(strict_reference_scores)),
            "reference_token_precision_mean": (
                _mean(
                    [
                        float(result["reference_token_precision"])
                        for result in answered_answerable
                        if result.get("reference_token_precision") is not None
                    ]
                )
            ),
            "reference_token_recall_mean": (
                _mean(
                    [
                        float(result["reference_token_recall"])
                        for result in answered_answerable
                        if result.get("reference_token_recall") is not None
                    ]
                )
            ),
        },
        "latency_ms": {
            **_distribution(
                _numeric_values(
                    results,
                    "context_latency_ms",
                ),
                prefix="context",
            ),
            **_distribution(
                _numeric_values(
                    results,
                    "generation_latency_ms",
                ),
                prefix="generation",
            ),
            **_distribution(
                _numeric_values(
                    results,
                    "total_latency_ms",
                ),
                prefix="total",
            ),
        },
    }


def flatten_numeric_summary(
    value: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, float]:
    flattened: dict[str, float] = {}

    for key, item in value.items():
        name = f"{prefix}_{key}" if prefix else key

        if isinstance(item, bool):
            continue

        if isinstance(item, (int, float)):
            if isfinite(float(item)):
                flattened[name] = float(item)
            continue

        if isinstance(item, Mapping):
            flattened.update(
                flatten_numeric_summary(
                    item,
                    prefix=name,
                )
            )

    return flattened


def _word_tokens(text: str) -> list[str]:
    normalized = normalize_reference_text(text)
    return re.findall(r"\w+", normalized)


def _count(
    results: Sequence[Mapping[str, Any]],
    *,
    decision: str,
    status: str,
) -> int:
    return sum(
        result.get("decision") == decision and result.get("status") == status for result in results
    )


def _reference_status(result: Mapping[str, Any]) -> str:
    status = result.get("benchmark_reference_status")

    if isinstance(status, str):
        return status

    # Compatibility with techqa_rag_eval_v1 artifacts.
    return (
        REFERENCE_AVAILABLE
        if result.get("answerability") == "answerable"
        else REFERENCE_MISSING
    )


def _generated_decision(result: Mapping[str, Any]) -> str | None:
    decision = result.get("parsed_decision")

    if decision is None:
        decision = result.get("decision")

    return str(decision) if decision is not None else None


def _generated_answer(result: Mapping[str, Any]) -> str | None:
    answer = result.get("parsed_answer")

    if answer is None:
        answer = result.get("answer")

    return str(answer) if answer is not None else None


def _has_output_contract_violation(diagnostics: Mapping[str, bool | int]) -> bool:
    return any(
        bool(diagnostics[key])
        for key in (
            "answer_source_id_leak",
            "answer_embedded_citation_list",
            "answer_over_120_words",
            "decision_content_mismatch",
        )
    )


def _strict_contract_valid(result: Mapping[str, Any]) -> bool:
    value = result.get("strict_contract_valid")

    if value is None:
        # Compatibility with successful evaluation artifacts written before repairs.
        return True

    return value is True


def _safe_divide(num: int | float, denom: int | float) -> float:
    if denom == 0:
        return 0.0

    return float(num) / float(denom)


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0

    return 2.0 * precision * recall / (precision + recall)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def _mean_booleans(values: Iterable[bool]) -> float:
    items = list(values)
    if not items:
        return 0.0

    return sum(bool(value) for value in items) / len(items)


def _numeric_values(
    results: Sequence[Mapping[str, Any]],
    key: str,
) -> list[float]:
    return [float(result[key]) for result in results if result.get(key) is not None]


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
        f"{prefix}_p50": _percentile(
            values,
            0.50,
        ),
        f"{prefix}_p95": _percentile(
            values,
            0.95,
        ),
    }


def _percentile(
    values: Sequence[float],
    p: float,
) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty values")

    if not 0.0 <= p <= 1.0:
        raise ValueError("percentile 'p' must be between 0 and 1")

    ordered_values = sorted(values)

    if len(ordered_values) == 1:
        return float(ordered_values[0])

    position = p * (len(ordered_values) - 1)
    lower_index = floor(position)
    upper_index = ceil(position)

    if lower_index == upper_index:
        return float(ordered_values[lower_index])

    integer_part = int(position)
    fractional_part = position - integer_part
    return ordered_values[integer_part] + fractional_part * (
        ordered_values[integer_part + 1] - ordered_values[integer_part]
    )
