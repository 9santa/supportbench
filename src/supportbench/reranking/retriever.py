from collections.abc import Callable, Sequence

from supportbench.data.models import Document
from supportbench.reranking.base import (
    RerankCandidate,
    Reranker,
)
from supportbench.retrieval.base import (
    Retriever,
    SearchResult,
)


type DocumentFormatter = Callable[
    [Document],
    str,
]


def format_document_for_reranking(document: Document) -> str:
    return f"{document.title}\n{document.text}"


class RerankingRetriever(Retriever):
    def __init__(
        self,
        *,
        candidate_retriever: Retriever,
        reranker: Reranker,
        documents: Sequence[Document],
        candidate_k: int = 50,
        document_formatter: DocumentFormatter = format_document_for_reranking,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")

        documents_by_id: dict[str, Document] = {}

        for document in documents:
            if document.doc_id in documents_by_id:
                raise ValueError("documents contain duplicate doc_id:", repr(document.doc_id))

            documents_by_id[document.doc_id] = document

        if not documents_by_id:
            raise ValueError("documents must not be empty")

        self._candidate_retriever = candidate_retriever
        self._reranker = reranker
        self._documents_by_id = documents_by_id
        self._candidate_k = candidate_k
        self._document_formatter = document_formatter

    @property
    def candidate_k(self) -> int:
        return self._candidate_k

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if top_k > self._candidate_k:
            raise ValueError("top_k must not be greater than candidate_k")

        if not query.strip():
            return []

        retrieved = self._candidate_retriever.search(
            query,
            top_k=self._candidate_k,
        )

        candidates = self._build_candidates(retrieved)

        reranked = self._reranker.rerank(
            query,
            candidates,
            top_k=top_k,
        )

        self._validate_reranker_results(
            reranked_doc_ids=[result.doc_id for result in reranked],
            candidate_doc_ids={candidate.doc_id for candidate in candidates},
        )

        return [
            SearchResult(
                doc_id=result.doc_id,
                score=result.score,
                rank=rank,
            )
            for rank, result in enumerate(reranked, start=1)
        ]

    def _build_candidates(
        self,
        retrieved: Sequence[SearchResult],
    ) -> tuple[RerankCandidate, ...]:
        candidates: list[RerankCandidate] = []
        seen_doc_ids: set[str] = set()

        for result in retrieved:
            if result.doc_id in seen_doc_ids:
                continue

            seen_doc_ids.add(result.doc_id)

            document = self._documents_by_id.get(result.doc_id)

            if document is None:
                raise ValueError(
                    f"candidate retriever returned an unknown document: {result.doc_id!r}"
                )

            candidates.append(
                RerankCandidate(
                    doc_id=result.doc_id,
                    text=self._document_formatter(document),
                    retrieval_score=result.score,
                    retrieval_rank=result.rank,
                )
            )

        return tuple(candidates)

    @staticmethod
    def _validate_reranker_results(
        *,
        reranked_doc_ids: Sequence[str],
        candidate_doc_ids: set[str],
    ) -> None:
        if len(reranked_doc_ids) != len(set(reranked_doc_ids)):
            raise ValueError("reranker returned duplcate document IDs")

        unknown_doc_ids = set(reranked_doc_ids) - candidate_doc_ids

        if unknown_doc_ids:
            formatted = ", ".join(sorted(unknown_doc_ids))

            raise ValueError(f"reranker returned documents outside the candidate set: {formatted}")
