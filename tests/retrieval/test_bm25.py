import pytest

from supportbench.data.models import Document
from supportbench.retrieval.base import SearchResult
from supportbench.retrieval.bm25 import BM25Retriever
from supportbench.retrieval.inverted_index import InvertedIndex


def _build_retriever(
    documents: list[Document],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> BM25Retriever:
    index = InvertedIndex.build(documents)

    return BM25Retriever(
        index,
        k1=k1,
        b=b,
    )


def _scores_by_document(
    results: list[SearchResult],
) -> dict[str, float]:
    return {result.doc_id: result.score for result in results}


@pytest.fixture
def retriever() -> BM25Retriever:
    documents = [
        Document(
            doc_id="vpn_ubuntu",
            title="VPN Ubuntu",
            text="Настройка корпоративного подключения.",
            category="network",
        ),
        Document(
            doc_id="vpn_windows",
            title="VPN Windows",
            text="Настройка корпоративного подключения.",
            category="network",
        ),
        Document(
            doc_id="ubuntu_network",
            title="Ubuntu Linux",
            text="Руководство по настройке сети.",
            category="network",
        ),
        Document(
            doc_id="gitlab_2fa",
            title="GitLab 2FA",
            text="Восстановление резервных кодов.",
            category="access",
        ),
    ]

    return _build_retriever(documents)


def test_exact_match_is_ranked_first(
    retriever: BM25Retriever,
) -> None:
    results = retriever.search("vpn ubuntu")

    assert results
    assert results[0].doc_id == "vpn_ubuntu"


def test_rare_term_has_more_influence() -> None:
    documents = [
        Document(
            doc_id="rare_document",
            title="Документ",
            text="vpn kubernetes",
            category="test",
        ),
        Document(
            doc_id="common_document_a",
            title="Документ",
            text="vpn настройка",
            category="test",
        ),
        Document(
            doc_id="common_document_b",
            title="Документ",
            text="vpn подключение",
            category="test",
        ),
    ]

    retriever = _build_retriever(documents)

    common_results = retriever.search("vpn")
    rare_results = retriever.search("kubernetes")

    common_scores = _scores_by_document(common_results)
    rare_scores = _scores_by_document(rare_results)

    assert rare_scores["rare_document"] > common_scores["rare_document"]


def test_repeated_document_term_has_saturating_gain() -> None:
    documents = [
        Document(
            doc_id="d1",
            title="Документ",
            text="vpn",
            category="test",
        ),
        Document(
            doc_id="d2",
            title="Документ",
            text="vpn vpn",
            category="test",
        ),
        Document(
            doc_id="d3",
            title="Документ",
            text="vpn vpn vpn vpn vpn vpn vpn vpn",
            category="test",
        ),
    ]

    # Disable length normalization (b=0) to test only TF saturation
    retriever = _build_retriever(documents, b=0.0)
    scores = _scores_by_document(retriever.search("vpn"))

    assert scores["d3"] > scores["d2"] > scores["d1"]
    assert scores["d3"] < scores["d1"] * 8


def test_shorter_document_is_preferred_when_other_factors_match() -> None:
    documents = [
        Document(
            doc_id="short",
            title="Документ",
            text="vpn настройка",
            category="test",
        ),
        Document(
            doc_id="long",
            title="Документ",
            text=(
                "vpn настройка плюс очень много посторонних "
                "слов про оборудование пользователей офис "
                "сервер приложение систему документацию"
            ),
            category="test",
        ),
    ]

    retriever = _build_retriever(documents, b=0.75)
    results = retriever.search("vpn настройка")

    assert [result.doc_id for result in results] == [
        "short",
        "long",
    ]
    assert results[0].score > results[1].score


def test_b_zero_disables_length_normalization() -> None:
    documents = [
        Document(
            doc_id="short",
            title="Документ",
            text="vpn настройка",
            category="test",
        ),
        Document(
            doc_id="long",
            title="Документ",
            text=(
                "vpn настройка плюс очень много посторонних "
                "слов про оборудование пользователей офис "
                "сервер приложение систему документацию"
            ),
            category="test",
        ),
    ]

    retriever = _build_retriever(documents, b=0.0)
    scores = _scores_by_document(retriever.search("vpn настройка"))

    assert scores["short"] == pytest.approx(scores["long"])


def test_empty_query_returns_no_results(
    retriever: BM25Retriever,
) -> None:
    assert retriever.search("") == []
    assert retriever.search("   ") == []
    assert retriever.search("--- !!!") == []


def test_unknown_query_returns_no_results(
    retriever: BM25Retriever,
) -> None:
    assert retriever.search("несуществующийтермин") == []


def test_top_k_limits_results(
    retriever: BM25Retriever,
) -> None:
    results = retriever.search(
        "vpn ubuntu настройка",
        top_k=2,
    )

    assert len(results) == 2


@pytest.mark.parametrize("top_k", [0, -1, -10])
def test_invalid_top_k_is_rejected(
    retriever: BM25Retriever,
    top_k: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="top_k must be positive",
    ):
        retriever.search("vpn", top_k=top_k)


@pytest.mark.parametrize("k1", [0.0, -0.1, -1.0])
def test_invalid_k1_is_rejected(k1: float) -> None:
    index = InvertedIndex.build(
        [
            Document(
                doc_id="document",
                title="VPN",
                text="Настройка",
                category="test",
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="k1 must be positive",
    ):
        BM25Retriever(index, k1=k1)


@pytest.mark.parametrize("b", [-0.1, -1.0, 1.1, 2.0])
def test_invalid_b_is_rejected(b: float) -> None:
    index = InvertedIndex.build(
        [
            Document(
                doc_id="document",
                title="VPN",
                text="Настройка",
                category="test",
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="b must be between 0 and 1",
    ):
        BM25Retriever(index, b=b)


def test_query_is_case_insensitive(
    retriever: BM25Retriever,
) -> None:
    lowercase_results = retriever.search("vpn ubuntu")
    uppercase_results = retriever.search("VPN UBUNTU")

    assert uppercase_results == lowercase_results


def test_equal_scores_are_ordered_by_doc_id() -> None:
    documents = [
        Document(
            doc_id="document_b",
            title="VPN настройка",
            text="Корпоративное подключение",
            category="test",
        ),
        Document(
            doc_id="document_a",
            title="VPN настройка",
            text="Корпоративное подключение",
            category="test",
        ),
    ]

    retriever = _build_retriever(documents)
    results = retriever.search("vpn настройка")

    assert [result.doc_id for result in results] == [
        "document_a",
        "document_b",
    ]
    assert results[0].score == pytest.approx(results[1].score)


def test_ranks_are_consecutive(
    retriever: BM25Retriever,
) -> None:
    results = retriever.search(
        "vpn ubuntu настройка gitlab",
    )

    assert [result.rank for result in results] == list(range(1, len(results) + 1))


def test_scores_are_positive(
    retriever: BM25Retriever,
) -> None:
    results = retriever.search(
        "vpn ubuntu настройка gitlab",
    )

    assert results
    assert all(result.score > 0.0 for result in results)


def test_repeated_query_terms_are_counted_once(
    retriever: BM25Retriever,
) -> None:
    unique_results = retriever.search("vpn ubuntu")
    repeated_results = retriever.search("vpn vpn vpn ubuntu")

    assert repeated_results == unique_results
