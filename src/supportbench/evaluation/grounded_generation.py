import time
from typing import Protocol

from supportbench.evaluation.rag_evaluator import (
    lexical_token_scores,
    output_contract_diagnostics,
)
from supportbench.rag.citations import (
    CitationContractError,
    CitationResolutionError,
    CitationValidationError,
)
from supportbench.rag.generation.models import GeneratedAnswer, LLMResponse
from supportbench.rag.generation.ollama import OllamaClientError
from supportbench.rag.generation.parser import GeneratedAnswerParseError
from supportbench.rag.generation.service import (
    GenerationTruncatedError,
    GroundedGenerationRun,
)
from supportbench.rag.models import RAGContext


class GroundedGenerator(Protocol):
    def generate(
        self,
        *,
        query: str,
        context: RAGContext,
    ) -> GroundedGenerationRun: ...


def evaluate_grounded_generation(
    *,
    query: str,
    reference_answer: str | None,
    relevant_doc_ids: tuple[str, ...],
    context: RAGContext,
    generator: GroundedGenerator,
    retries: int,
) -> dict[str, object]:
    if retries < 0:
        raise ValueError("retries must be non-negative")

    started = time.perf_counter()
    final_client_error: OllamaClientError | None = None

    for attempt in range(retries + 1):
        try:
            generation = generator.generate(query=query, context=context)
            return _success_result(
                generation=generation,
                reference_answer=reference_answer,
                relevant_doc_ids=relevant_doc_ids,
                elapsed_ms=_elapsed_ms(started),
            )
        except OllamaClientError as error:
            final_client_error = error

            if attempt < retries:
                time.sleep(min(2.0**attempt, 8.0))
                continue

        except GenerationTruncatedError as error:
            return _error_result(
                status="generation_truncated",
                error=error,
                raw_response=error.raw_response,
                llm_response=error.llm_response,
                parsed_answer=None,
                raw_citation_ids=(),
                resolved_citation_ids=(),
                elapsed_ms=_elapsed_ms(started),
            )
        except GeneratedAnswerParseError as error:
            return _error_result(
                status="parse_error",
                error=error,
                raw_response=error.raw_response,
                llm_response=error.llm_response,
                parsed_answer=None,
                raw_citation_ids=(),
                resolved_citation_ids=(),
                elapsed_ms=_elapsed_ms(started),
            )
        except CitationResolutionError as error:
            return _error_result(
                status="citation_resolution_error",
                error=error,
                raw_response=error.raw_response,
                llm_response=error.llm_response,
                parsed_answer=error.parsed_answer,
                raw_citation_ids=error.raw_citation_ids,
                resolved_citation_ids=error.citation_ids,
                elapsed_ms=_elapsed_ms(started),
            )
        except CitationContractError as error:
            return _error_result(
                status="citation_contract_error",
                error=error,
                raw_response=error.raw_response,
                llm_response=error.llm_response,
                parsed_answer=error.parsed_answer,
                raw_citation_ids=error.raw_citation_ids,
                resolved_citation_ids=error.citation_ids,
                elapsed_ms=_elapsed_ms(started),
            )
        except CitationValidationError as error:
            return _error_result(
                status="citation_error",
                error=error,
                raw_response=error.raw_response,
                llm_response=error.llm_response,
                parsed_answer=error.parsed_answer,
                raw_citation_ids=error.raw_citation_ids,
                resolved_citation_ids=error.citation_ids,
                elapsed_ms=_elapsed_ms(started),
            )

    assert final_client_error is not None
    return _error_result(
        status="llm_error",
        error=final_client_error,
        raw_response=None,
        llm_response=None,
        parsed_answer=None,
        raw_citation_ids=(),
        resolved_citation_ids=(),
        elapsed_ms=_elapsed_ms(started),
    )


def _success_result(
    *,
    generation: GroundedGenerationRun,
    reference_answer: str | None,
    relevant_doc_ids: tuple[str, ...],
    elapsed_ms: float,
) -> dict[str, object]:
    answer = generation.answer
    diagnostics = output_contract_diagnostics(
        decision=answer.decision,
        answer=answer.answer,
    )
    precision, recall, f1 = (
        lexical_token_scores(answer.answer, reference_answer)
        if answer.decision == "answer"
        else (None, None, None)
    )

    return {
        "status": "success",
        "parsed_decision": answer.decision,
        "parsed_answer": answer.answer,
        "decision": answer.decision,
        "answer": answer.answer,
        "citation_ids": list(answer.citation_ids),
        "raw_citation_ids": list(generation.raw_citation_ids),
        "resolved_citation_ids": list(generation.resolved_citation_ids),
        "contract_repaired": generation.contract_repaired,
        "strict_contract_valid": generation.strict_contract_valid,
        "contract_violations": list(generation.contract_violations),
        **diagnostics,
        "full_output_contract_valid": (
            generation.strict_contract_valid and not _has_output_contract_violation(diagnostics)
        ),
        "raw_response": generation.raw_response,
        "llm_called": generation.raw_response is not None,
        **_llm_metadata(generation.llm_response),
        "gold_document_cited": bool(set(answer.citation_ids) & set(relevant_doc_ids))
        if answer.decision == "answer"
        else None,
        "reference_token_precision": precision,
        "reference_token_recall": recall,
        "reference_token_f1": f1,
        "generation_latency_ms": elapsed_ms,
        "total_latency_ms": elapsed_ms,
        "error_type": None,
        "error_message": None,
    }


def _error_result(
    *,
    status: str,
    error: Exception,
    raw_response: str | None,
    llm_response: LLMResponse | None,
    parsed_answer: GeneratedAnswer | None,
    raw_citation_ids: tuple[str, ...],
    resolved_citation_ids: tuple[str, ...],
    elapsed_ms: float,
) -> dict[str, object]:
    parsed_decision = parsed_answer.decision if parsed_answer is not None else None
    parsed_answer_text = parsed_answer.answer if parsed_answer is not None else None
    diagnostics = output_contract_diagnostics(
        decision=parsed_decision,
        answer=parsed_answer_text,
    )

    return {
        "status": status,
        "parsed_decision": parsed_decision,
        "parsed_answer": parsed_answer_text,
        "decision": None,
        "answer": None,
        "citation_ids": [],
        "raw_citation_ids": list(raw_citation_ids),
        "resolved_citation_ids": list(resolved_citation_ids),
        "contract_repaired": False,
        "strict_contract_valid": False if status == "citation_contract_error" else None,
        "contract_violations": list(error.contract_violations)
        if isinstance(error, CitationValidationError)
        else [],
        **diagnostics,
        "full_output_contract_valid": False,
        "raw_response": raw_response,
        "llm_called": True,
        **_llm_metadata(llm_response),
        "gold_document_cited": None,
        "reference_token_precision": None,
        "reference_token_recall": None,
        "reference_token_f1": None,
        "generation_latency_ms": elapsed_ms,
        "total_latency_ms": elapsed_ms,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }


def _llm_metadata(response: LLMResponse | None) -> dict[str, object]:
    return {
        "done_reason": response.done_reason if response is not None else None,
        "prompt_eval_count": response.prompt_eval_count if response is not None else None,
        "eval_count": response.eval_count if response is not None else None,
    }


def _has_output_contract_violation(diagnostics: dict[str, bool | int]) -> bool:
    return any(
        bool(diagnostics[key])
        for key in (
            "answer_source_id_leak",
            "answer_embedded_citation_list",
            "answer_over_120_words",
            "decision_content_mismatch",
        )
    )


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1_000.0
