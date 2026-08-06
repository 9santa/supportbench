import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Literal

from supportbench.chunking.models import Chunk
from supportbench.rag.context_builder import RepresentativeChunkContextBuilder
from supportbench.rag.document_store import DocumentStore
from supportbench.rag.generation.prompt import PromptBudget, PromptBudgetCalculator
from supportbench.rag.models import RAGContext, RetrievedChunk
from supportbench.rag.retrieval import ParentRetrievalRun, ParentRetrievalService
from supportbench.reranking.base import RerankCandidate, Reranker, RerankResult
from supportbench.reranking.retriever import format_document_for_reranking
from supportbench.retrieval.base import SearchResult

type EvidenceSelection = Literal[
    "retrieval_representatives",
    "within_parent_rerank",
]


@dataclass(frozen=True, slots=True)
class ContextPreparationRun:
    retrieval: ParentRetrievalRun
    retrieved_chunks: tuple[RetrievedChunk, ...]
    context: RAGContext
    prompt_budget: PromptBudget | None = None
    prompt_token_count: int = 0


class RepresentativeChunkResolver:
    """Select and materialize generation evidence for final parent results."""

    def __init__(
        self,
        *,
        chunk_store: DocumentStore,
        chunks_by_id: Mapping[str, Chunk],
        reranker: Reranker | None = None,
        chunks_per_parent: int = 2,
        evidence_selection: EvidenceSelection = "retrieval_representatives",
    ) -> None:
        if not chunks_by_id:
            raise ValueError("chunks_by_id must not be empty")

        if chunks_per_parent <= 0:
            raise ValueError("chunks_per_parent must be positive")

        if evidence_selection not in (
            "retrieval_representatives",
            "within_parent_rerank",
        ):
            raise ValueError(f"unknown evidence selection: {evidence_selection!r}")

        if evidence_selection == "within_parent_rerank" and reranker is None:
            raise ValueError("within-parent evidence selection requires a reranker")

        self._chunk_store = chunk_store
        self._chunks_by_id = dict(chunks_by_id)
        self._reranker = reranker
        self._chunks_per_parent = chunks_per_parent
        self._evidence_selection = evidence_selection
        chunks_by_parent: dict[str, list[str]] = {}

        for chunk in sorted(
            chunks_by_id.values(),
            key=lambda item: (item.document_id, item.ordinal, item.chunk_id),
        ):
            chunks_by_parent.setdefault(chunk.document_id, []).append(chunk.chunk_id)

        self._chunk_ids_by_parent = {
            parent_id: tuple(chunk_ids)
            for parent_id, chunk_ids in chunks_by_parent.items()
        }

    def resolve(
        self,
        run: ParentRetrievalRun,
        *,
        query: str | None = None,
        top_k: int = 5,
        evidence_selection: EvidenceSelection | None = None,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if not run.fused_parents:
            return []

        parent_results = run.fused_parents[:top_k]
        self._validate_parent_results(parent_results)
        selected_strategy = evidence_selection or self._evidence_selection

        if selected_strategy not in (
            "retrieval_representatives",
            "within_parent_rerank",
        ):
            raise ValueError(f"unknown evidence selection: {selected_strategy!r}")

        chunk_ids_by_parent = self._select_chunk_ids(
            query=query,
            run=run,
            parent_results=parent_results,
            evidence_selection=selected_strategy,
        )
        retrieved_chunks: list[RetrievedChunk] = []
        seen_chunk_ids: set[str] = set()

        for parent_result in parent_results:
            chunk_ids = chunk_ids_by_parent.get(parent_result.doc_id)

            if chunk_ids is None:
                raise ValueError(
                    "final parent retriever returned a parent without evidence "
                    f"chunks: {parent_result.doc_id!r}"
                )

            for evidence_rank, chunk_id in enumerate(chunk_ids, start=1):
                if chunk_id in seen_chunk_ids:
                    raise ValueError(f"duplicate representative chunk ID: {chunk_id!r}")

                metadata = self._chunks_by_id.get(chunk_id)

                if metadata is None:
                    raise ValueError(f"unknown representative chunk ID: {chunk_id!r}")

                if metadata.document_id != parent_result.doc_id:
                    raise ValueError(
                        f"chunk {chunk_id!r} belongs to {metadata.document_id!r}, "
                        f"not {parent_result.doc_id!r}"
                    )

                try:
                    document = self._chunk_store.get(chunk_id)
                except KeyError as error:
                    raise ValueError(
                        f"representative chunk is missing from the runtime corpus: {chunk_id!r}"
                    ) from error

                if document.text != metadata.text:
                    raise ValueError(
                        f"runtime text and chunk metadata differ for {chunk_id!r}"
                    )

                seen_chunk_ids.add(chunk_id)
                retrieved_chunks.append(
                    RetrievedChunk(
                        chunk_id=chunk_id,
                        parent_doc_id=parent_result.doc_id,
                        document_title=metadata.document_title,
                        text=metadata.text,
                        category=document.category,
                        section_path=metadata.section_path,
                        ordinal=metadata.ordinal,
                        start_char=metadata.start_char,
                        end_char=metadata.end_char,
                        parent_score=float(parent_result.score),
                        parent_rank=parent_result.rank,
                        evidence_rank=evidence_rank,
                    )
                )

        return retrieved_chunks

    def _select_chunk_ids(
        self,
        *,
        query: str | None,
        run: ParentRetrievalRun,
        parent_results: Sequence[SearchResult],
        evidence_selection: EvidenceSelection,
    ) -> Mapping[str, tuple[str, ...]]:
        if evidence_selection == "retrieval_representatives":
            return run.representative_chunks_by_parent

        if query is None or not query.strip():
            raise ValueError("query must be non-empty for within-parent evidence reranking")

        reranker = self._reranker
        assert reranker is not None

        parent_scores = {result.doc_id: float(result.score) for result in parent_results}
        candidate_chunk_ids: list[str] = []

        for result in parent_results:
            chunk_ids = self._chunk_ids_by_parent.get(result.doc_id)

            if not chunk_ids:
                raise ValueError(
                    f"final parent has no chunks in the runtime corpus: {result.doc_id!r}"
                )

            candidate_chunk_ids.extend(chunk_ids)

        try:
            documents = self._chunk_store.get_many(candidate_chunk_ids)
        except KeyError as error:
            raise ValueError("within-parent chunk is missing from the runtime corpus") from error

        candidates: list[RerankCandidate] = []

        for retrieval_rank, (chunk_id, document) in enumerate(
            zip(candidate_chunk_ids, documents, strict=True),
            start=1,
        ):
            metadata = self._chunks_by_id[chunk_id]

            if document.doc_id != chunk_id or document.text != metadata.text:
                raise ValueError(
                    f"runtime document and chunk metadata differ for {chunk_id!r}"
                )

            candidates.append(
                RerankCandidate(
                    doc_id=chunk_id,
                    text=format_document_for_reranking(document),
                    retrieval_score=parent_scores[metadata.document_id],
                    retrieval_rank=retrieval_rank,
                )
            )

        reranked = reranker.rerank(
            query,
            candidates,
            top_k=len(candidates),
        )
        self._validate_reranker_results(
            reranked,
            candidate_ids=set(candidate_chunk_ids),
        )
        selected: dict[str, list[str]] = {
            result.doc_id: [] for result in parent_results
        }

        for reranked_chunk in reranked:
            parent_id = self._chunks_by_id[reranked_chunk.doc_id].document_id
            parent_chunks = selected[parent_id]

            if len(parent_chunks) < self._chunks_per_parent:
                parent_chunks.append(reranked_chunk.doc_id)

        return {
            parent_id: tuple(chunk_ids)
            for parent_id, chunk_ids in selected.items()
        }

    @staticmethod
    def _validate_parent_results(results: Sequence[SearchResult]) -> None:
        seen_parent_ids: set[str] = set()

        for expected_rank, result in enumerate(results, start=1):
            if not result.doc_id.strip():
                raise ValueError("parent retriever returned an empty document ID")

            if result.doc_id in seen_parent_ids:
                raise ValueError(f"parent retriever returned duplicate ID: {result.doc_id!r}")

            if result.rank != expected_rank:
                raise ValueError(
                    "parent ranks must be consecutive starting at 1; "
                    f"expected {expected_rank}, received {result.rank}"
                )

            if not isinstance(result.score, Real) or not math.isfinite(float(result.score)):
                raise ValueError("parent score must be finite")

            seen_parent_ids.add(result.doc_id)

    @staticmethod
    def _validate_reranker_results(
        results: Sequence[RerankResult],
        *,
        candidate_ids: set[str],
    ) -> None:
        result_ids = [result.doc_id for result in results]

        if len(result_ids) != len(set(result_ids)):
            raise ValueError("evidence reranker returned duplicate chunk IDs")

        if set(result_ids) != candidate_ids:
            raise ValueError("evidence reranker must return every candidate chunk exactly once")


class ContextPreparationService:
    """Retrieve evidence and construct a generation-ready context for one query."""

    def __init__(
        self,
        *,
        retrieval_service: ParentRetrievalService,
        chunk_resolver: RepresentativeChunkResolver,
        context_builder: RepresentativeChunkContextBuilder,
        prompt_budget_calculator: PromptBudgetCalculator,
        top_parents: int,
    ) -> None:
        if top_parents <= 0:
            raise ValueError("top_parents must be positive")

        self._retrieval_service = retrieval_service
        self._chunk_resolver = chunk_resolver
        self._context_builder = context_builder
        self._prompt_budget_calculator = prompt_budget_calculator
        self._top_parents = top_parents

    def prepare(
        self,
        query: str,
        *,
        retrieval: ParentRetrievalRun | None = None,
        evidence_selection: EvidenceSelection | None = None,
    ) -> ContextPreparationRun:
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("query must be non-empty")

        prompt_budget = self._prompt_budget_calculator.calculate(normalized_query)
        retrieval_run = retrieval or self._retrieval_service.retrieve(normalized_query)
        retrieved_chunks = tuple(
            self._chunk_resolver.resolve(
                retrieval_run,
                query=normalized_query,
                top_k=self._top_parents,
                evidence_selection=evidence_selection,
            )
        )
        context_token_budget = prompt_budget.available_context_tokens
        context = self._context_builder.build(
            retrieved_chunks,
            max_tokens=context_token_budget,
        )
        prompt_token_count = self._prompt_budget_calculator.count_prompt(
            query=normalized_query,
            context=context,
        )
        overflow = (
            prompt_token_count
            + prompt_budget.reserved_output_tokens
            - prompt_budget.model_context_window
        )

        if overflow > 0:
            context_token_budget -= overflow

            if context_token_budget <= 0:
                raise ValueError("full prompt leaves no room for knowledge context")

            context = self._context_builder.build(
                retrieved_chunks,
                max_tokens=context_token_budget,
            )
            prompt_token_count = self._prompt_budget_calculator.count_prompt(
                query=normalized_query,
                context=context,
            )

        if (
            prompt_token_count + prompt_budget.reserved_output_tokens
            > prompt_budget.model_context_window
        ):
            raise RuntimeError("full prompt exceeded the model context window")

        return ContextPreparationRun(
            retrieval=retrieval_run,
            retrieved_chunks=retrieved_chunks,
            context=context,
            prompt_budget=prompt_budget,
            prompt_token_count=prompt_token_count,
        )
