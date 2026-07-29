import math
from numbers import Real

from supportbench.rag.document_store import (
    DocumentStore,
)
from supportbench.rag.models import (
    RetrievedDocument,
)
from supportbench.retrieval.base import Retriever

# RetrievalPipeline doesn't sort results, only verifies.
# If retriever returned wrong ranks -> it is his problem.


class RetrievalPipeline:
    def __init__(
        self,
        *,
        retriever: Retriever,
        document_store: DocumentStore,
    ) -> None:
        self._retriever = retriever
        self._document_store = document_store

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[RetrievedDocument]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if not query.strip():
            return []

        search_results = self._retriever.search(
            query,
            top_k=top_k,
        )

        retrieved_documents: list[RetrievedDocument] = []

        seen_doc_ids: set[str] = set()

        for expected_rank, result in enumerate(search_results, start=1):
            self._validate_result(
                doc_id=result.doc_id,
                score=result.score,
                rank=result.rank,
                expected_rank=expected_rank,
                seen_doc_ids=seen_doc_ids,
            )

            try:
                document = self._document_store.get(result.doc_id)
            except KeyError as error:
                raise ValueError(
                    f"retriever returned an unknown document: {result.doc_id!r}"
                ) from error

            seen_doc_ids.add(result.doc_id)

            retrieved_documents.append(
                RetrievedDocument(
                    doc_id=document.doc_id,
                    title=document.title,
                    text=document.text,
                    category=document.category,
                    score=float(result.score),
                    rank=result.rank,
                )
            )

        return retrieved_documents

    @staticmethod
    def _validate_result(
        *,
        doc_id: object,
        score: object,
        rank: object,
        expected_rank: int,
        seen_doc_ids: set[str],
    ) -> None:
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError(f"retriever returned an invalid document ID: {doc_id!r}")

        if doc_id in seen_doc_ids:
            raise ValueError(f"retriever returned duplicate document ID: {doc_id!r}")

        if not isinstance(rank, int) or rank < 1:
            raise ValueError("retriever rank must be a positive integer")

        if rank != expected_rank:
            raise ValueError(
                "retriever ranks must be "
                "consecutive starting at 1; "
                f"expected {expected_rank}, "
                f"received {rank}"
            )

        if not isinstance(score, Real):
            raise ValueError("retriever score must be numeric")

        if not math.isfinite(float(score)):
            raise ValueError("retriever score must be finite")
