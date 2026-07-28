from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from supportbench.data.models import Document
from supportbench.retrieval.bm25 import BM25Retriever
from supportbench.retrieval.factory import RetrieverConfig, RetrieverFactory
from supportbench.retrieval.inverted_index import InvertedIndex
from supportbench.retrieval.tfidf import TfidfRetriever


def _documents() -> list[Document]:
    return [
        Document(
            doc_id="vpn_setup",
            title="VPN setup",
            text="Configure VPN on a workstation.",
            category="network",
        ),
        Document(
            doc_id="gitlab_recovery",
            title="GitLab recovery",
            text="Recover access to GitLab.",
            category="access",
        ),
    ]


def _config(
    *,
    dense_batch_size: int = 16,
    bm25_k1: float = 0.5,
    bm25_b: float = 1.0,
    candidate_k: int = 50,
    rrf_k: int = 60,
) -> RetrieverConfig:
    return RetrieverConfig(
        dense_index_path=Path("unused"),
        dense_model_name="test-model",
        dense_batch_size=dense_batch_size,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        candidate_k=candidate_k,
        rrf_k=rrf_k,
    )


def test_factory_reuses_inverted_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_count = 0
    original_build = InvertedIndex.build

    def counting_build(documents: Sequence[Document]) -> InvertedIndex:
        nonlocal build_count
        build_count += 1
        return original_build(documents)

    monkeypatch.setattr(InvertedIndex, "build", staticmethod(counting_build))

    factory = RetrieverFactory(_documents(), config=_config())

    assert isinstance(factory.create("tfidf"), TfidfRetriever)
    assert isinstance(factory.create("bm25"), BM25Retriever)
    assert build_count == 1


@pytest.mark.parametrize(
    ("create_config", "message"),
    [
        (lambda: _config(dense_batch_size=0), "dense_batch_size must be positive"),
        (lambda: _config(bm25_k1=0.0), "bm25_k1 must be finite and positive"),
        (lambda: _config(bm25_b=1.1), "bm25_b must be finite and between 0 and 1"),
        (lambda: _config(candidate_k=0), "candidate_k must be positive"),
        (lambda: _config(rrf_k=0), "rrf_k must be positive"),
    ],
)
def test_retriever_config_rejects_invalid_values(
    create_config: Callable[[], RetrieverConfig],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        create_config()
