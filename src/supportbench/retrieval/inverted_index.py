from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Self

from supportbench.data.models import Document
from supportbench.retrieval.tokenization import tokenize


@dataclass(frozen=True, slots=True)
class CorpusStats:
    document_count: int
    vocab_size: int
    avg_doc_len: float


class InvertedIndex:
    __slots__ = (
        "_postings",
        "_doc_lens",
        "_stats",
    )

    def __init__(
        self,
        *,
        postings: Mapping[str, Mapping[str, int]],
        doc_lens: Mapping[str, int],
    ) -> None:
        self._postings = MappingProxyType(
            {
                term: MappingProxyType(dict(term_postings))
                for term, term_postings in postings.items()
            }
        )
        self._doc_lens = MappingProxyType(dict(doc_lens))

        doc_count = len(self._doc_lens)
        total_doc_len = sum(self._doc_lens.values())

        avg_doc_len = total_doc_len / doc_count if doc_count > 0 else 0.0

        self._stats = CorpusStats(
            document_count=doc_count,
            vocab_size=len(self._postings),
            avg_doc_len=avg_doc_len,
        )

    @classmethod
    def build(cls, documents: list[Document]) -> Self:
        postings: dict[str, dict[str, int]] = {}
        doc_lens: dict[str, int] = {}

        for document in documents:
            doc_id = document.doc_id

            if not doc_id.strip():
                raise ValueError("document id must be non-empty")

            if doc_id in doc_lens:
                raise ValueError(f"duplicate document id: {doc_id!r}")

            tokens = tokenize(f"{document.title} {document.text}")

            if not tokens:
                raise ValueError(f"document {doc_id!r} contains no indexable tokens")

            doc_lens[doc_id] = len(tokens)
            term_counts = Counter(tokens)

            for term, freq in term_counts.items():
                postings.setdefault(term, {})[doc_id] = freq

        return cls(postings=postings, doc_lens=doc_lens)

    @property
    def statistics(self) -> CorpusStats:
        return self._stats

    @property
    def terms(self) -> tuple[str, ...]:
        return tuple(sorted(self._postings))

    @property
    def document_ids(self) -> tuple[str, ...]:
        return tuple(self._doc_lens)

    @property
    def document_count(self) -> int:
        return len(self._doc_lens)

    def term_frequency(self, term: str, doc_id: str) -> int:
        """Returns count of the term in document doc_id"""
        normalized_term = self._normalize_term(term)
        self._validate_doc_id(doc_id)

        # get(), not setdefault()
        postings = self._postings.get(normalized_term)
        if postings is None:
            return 0

        return postings.get(doc_id, 0)

    def document_frequency(self, term: str) -> int:
        """Returns how many documents include this term."""
        normalized_term = self._normalize_term(term)
        return len(self._postings.get(normalized_term, {}))

    def postings_for(self, term: str) -> Mapping[str, int]:
        normalized_term = self._normalize_term(term)
        return self._postings.get(normalized_term, MappingProxyType({}))

    def document_length(self, doc_id: str) -> int:
        self._validate_doc_id(doc_id)
        return self._doc_lens[doc_id]

    def _validate_doc_id(self, doc_id: str) -> None:
        if doc_id not in self._doc_lens:
            raise KeyError(f"unknown document id: {doc_id!r}")

    @staticmethod  # static because it doesn't use any class stuff
    def _normalize_term(term: str) -> str:
        tokens = tokenize(term)
        if len(tokens) != 1:
            raise ValueError("term must contain exactly one token")

        return tokens[0]
