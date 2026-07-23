import pytest

from supportbench.data.models import Document
from supportbench.retrieval.inverted_index import InvertedIndex
from supportbench.retrieval.tfidf import TfidfRetriever


@pytest.fixture
def documents() -> list[Document]:
    return [
        Document(
            doc_id="vpn_linux",
            title="VPN на Linux",
            text="Настройка VPN с помощью OpenVPN.",
            category="network",
        ),
        Document(
            doc_id="vpn_windows",
            title="VPN на Windows",
            text="Установите корпоративный VPN-клиент.",
            category="network",
        ),
        Document(
            doc_id="gitlab_2fa",
            title="Доступ к GitLab",
            text=("Восстановление двухфакторной аутентификации."),
            category="access",
        ),
    ]


@pytest.fixture
def index(
    documents: list[Document],
) -> InvertedIndex:
    return InvertedIndex.build(documents)


@pytest.fixture
def retriever(
    index: InvertedIndex,
) -> TfidfRetriever:
    return TfidfRetriever(index)


def test_search_returns_relevant_document_first(
    retriever: TfidfRetriever,
) -> None:
    results = retriever.search("OpenVPN Linux")

    assert results
    assert results[0].doc_id == "vpn_linux"


def test_search_results_have_descending_scores(
    retriever: TfidfRetriever,
) -> None:
    results = retriever.search("VPN настройка Windows")

    scores = [result.score for result in results]

    assert scores == sorted(scores, reverse=True)


def test_search_ranks_start_at_one(
    retriever: TfidfRetriever,
) -> None:
    results = retriever.search("VPN")

    assert [result.rank for result in results] == list(range(1, len(results) + 1))


def test_search_respects_top_k(
    retriever: TfidfRetriever,
) -> None:
    results = retriever.search("VPN", top_k=1)

    assert len(results) == 1
    assert results[0].rank == 1


def test_top_k_larger_than_candidate_count_returns_all(
    retriever: TfidfRetriever,
) -> None:
    results = retriever.search("VPN", top_k=100)

    assert len(results) == 2


@pytest.mark.parametrize("top_k", [0, -1, -10])
def test_non_positive_top_k_is_rejected(
    retriever: TfidfRetriever,
    top_k: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="top_k must be positive",
    ):
        retriever.search("VPN", top_k=top_k)


def test_empty_query_returns_no_results(
    retriever: TfidfRetriever,
) -> None:
    assert retriever.search("") == []


def test_punctuation_only_query_returns_no_results(
    retriever: TfidfRetriever,
) -> None:
    assert retriever.search("--- !!! ...") == []


def test_unknown_terms_return_no_results(
    retriever: TfidfRetriever,
) -> None:
    assert retriever.search("несуществующийтермин") == []


def test_unknown_terms_are_ignored(
    retriever: TfidfRetriever,
) -> None:
    known_results = retriever.search("VPN")
    mixed_results = retriever.search("VPN несуществующийтермин")

    assert mixed_results == known_results


def test_results_do_not_have_zero_scores(
    retriever: TfidfRetriever,
) -> None:
    results = retriever.search("VPN")

    assert all(result.score > 0.0 for result in results)


def test_query_is_case_normalized(
    retriever: TfidfRetriever,
) -> None:
    lowercase_results = retriever.search("vpn")
    uppercase_results = retriever.search("VPN")

    assert uppercase_results == lowercase_results


def test_equal_scores_are_sorted_by_document_id() -> None:
    documents = [
        Document(
            doc_id="document_b",
            title="Общий термин",
            text="Одинаковый текст",
            category="test",
        ),
        Document(
            doc_id="document_a",
            title="Общий термин",
            text="Одинаковый текст",
            category="test",
        ),
    ]

    index = InvertedIndex.build(documents)
    retriever = TfidfRetriever(index)

    results = retriever.search("общий термин одинаковый текст")

    assert [result.doc_id for result in results] == [
        "document_a",
        "document_b",
    ]
    assert results[0].score == pytest.approx(results[1].score)


def test_identical_query_and_document_have_unit_score() -> None:
    documents = [
        Document(
            doc_id="document",
            title="Alpha",
            text="Beta",
            category="test",
        ),
    ]

    index = InvertedIndex.build(documents)
    retriever = TfidfRetriever(index)

    results = retriever.search("alpha beta")

    assert len(results) == 1
    assert results[0].score == pytest.approx(1.0)


def test_repeated_query_term_increases_its_weight() -> None:
    documents = [
        Document(
            doc_id="alpha_document",
            title="Alpha",
            text="Common",
            category="test",
        ),
        Document(
            doc_id="beta_document",
            title="Beta",
            text="Common",
            category="test",
        ),
    ]

    index = InvertedIndex.build(documents)
    retriever = TfidfRetriever(index)

    balanced_results = retriever.search("alpha beta")
    repeated_results = retriever.search("alpha alpha beta")

    balanced_scores = {result.doc_id: result.score for result in balanced_results}
    repeated_scores = {result.doc_id: result.score for result in repeated_results}

    assert balanced_scores["alpha_document"] == pytest.approx(balanced_scores["beta_document"])

    assert repeated_scores["alpha_document"] > repeated_scores["beta_document"]
