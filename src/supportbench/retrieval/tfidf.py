from collections import Counter, defaultdict
from math import log, sqrt

from supportbench.retrieval.base import SearchResult
from supportbench.retrieval.inverted_index import InvertedIndex
from supportbench.retrieval.tokenization import tokenize


class TfidfRetriever:
    __slots__ = (
        "_index",
        "_idf",
        "_document_norms",
    )

    def __init__(self, index: InvertedIndex) -> None:
        self._index = index
        self._idf = self._compute_idf()
        self._document_norms = self._compute_document_norms()

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        query_counts = Counter(query_tokens)
        query_weights: dict[str, float] = {}

        for term, count in query_counts.items():
            idf = self._idf.get(term)

            if idf is None:
                continue

            query_tf = self._term_frequency(count)
            query_weights[term] = query_tf * idf

        if not query_weights:
            return []

        query_norm = sqrt(sum(weight * weight for weight in query_weights.values()))

        dot_products: defaultdict[str, float] = defaultdict(float)

        for term, query_weight in query_weights.items():
            idf = self._idf[term]

            for doc_id, count in self._index.postings_for(term).items():
                document_tf = self._term_frequency(count)
                document_weight = document_tf * idf

                dot_products[doc_id] += query_weight * document_weight

        scores: dict[str, float] = {}

        for doc_id, dot_product in dot_products.items():
            document_norm = self._document_norms[doc_id]

            if document_norm == 0.0:
                continue

            score = dot_product / (query_norm * document_norm)

            if score > 0.0:
                scores[doc_id] = score

        ordered_scores = sorted(scores.items(), key=lambda item: (-item[1], item[0]))

        return [
            SearchResult(doc_id=doc_id, score=score, rank=rank)
            for rank, (doc_id, score) in enumerate(ordered_scores[:top_k], start=1)
        ]

    def _compute_idf(self) -> dict[str, float]:
        document_count = self._index.document_count

        return {
            term: log((document_count + 1) / (self._index.document_frequency(term) + 1) + 1)
            for term in self._index.terms
        }

    def _compute_document_norms(self) -> dict[str, float]:
        squared_norms = {doc_id: 0.0 for doc_id in self._index.document_ids}

        for term, idf in self._idf.items():
            for doc_id, count in self._index.postings_for(term).items():
                term_frequency = self._term_frequency(count)
                weight = term_frequency * idf

                squared_norms[doc_id] += weight * weight

        return {doc_id: sqrt(squared_norm) for doc_id, squared_norm in squared_norms.items()}

    @staticmethod
    def _term_frequency(count: int) -> float:
        if count <= 0:
            return 0.0

        return 1.0 + log(count)
