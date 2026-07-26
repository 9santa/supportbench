from collections import defaultdict
from math import log

from supportbench.retrieval.base import SearchResult
from supportbench.retrieval.inverted_index import InvertedIndex
from supportbench.retrieval.tokenization import tokenize


class BM25Retriever:
    __slots__ = (
        "_index",
        "_k1",
        "_b",
        "_idf",
        "_avg_doc_len",
    )

    def __init__(
        self,
        index: InvertedIndex,
        *,
        k1: float = 0.5,
        b: float = 1.0,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")

        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be between 0 and 1")

        self._index = index
        self._k1 = k1
        self._b = b
        self._avg_doc_len = index.statistics.avg_doc_len
        self._idf = self._compute_idf()

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_terms = set(tokenize(query))

        if not query_terms:
            return []

        scores: defaultdict[str, float] = defaultdict(float)

        for term in query_terms:
            idf = self._idf.get(term)

            if idf is None:
                continue

            for doc_id, frequency in self._index.postings_for(term).items():
                document_length = self._index.document_length(doc_id)

                scores[doc_id] += idf * self._term_score(
                    frequency=frequency, document_length=document_length
                )

        ordered_scores = sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0]),
        )

        return [
            SearchResult(
                doc_id=doc_id,
                score=score,
                rank=rank,
            )
            for rank, (doc_id, score) in enumerate(ordered_scores[:top_k], start=1)
            if score > 0.0
        ]

    def _compute_idf(self) -> dict[str, float]:
        document_count = self._index.document_count
        idf: dict[str, float] = {}

        for term in self._index.terms:
            document_frequence = self._index.document_frequency(term)

            idf[term] = log(
                1.0 + (document_count - document_frequence + 0.5) / (document_frequence + 0.5)
            )

        return idf

    def _term_score(
        self,
        *,
        frequency: int,
        document_length: int,
    ) -> float:
        length_ratio = document_length / self._avg_doc_len

        normalization = self._k1 * (1.0 - self._b + self._b * length_ratio)

        return frequency * (self._k1 + 1.0) / (frequency + normalization)
