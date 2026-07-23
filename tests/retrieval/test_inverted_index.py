import pytest

from supportbench.data.models import Document
from supportbench.retrieval.inverted_index import InvertedIndex
from supportbench.retrieval.tokenization import tokenize


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
            text="Восстановление двухфакторной аутентификации.",
            category="access",
        ),
    ]


@pytest.fixture
def index(documents: list[Document]) -> InvertedIndex:
    return InvertedIndex.build(documents)


def test_build_index_counts_documents(index: InvertedIndex) -> None:
    assert index.statistics.document_count == 3


def test_build_index_counts_vocab(index: InvertedIndex) -> None:
    assert index.statistics.vocab_size == 17


def test_document_length_is_token_count(index: InvertedIndex, documents: list[Document]) -> None:
    doc = documents[0]

    expected_len = len(tokenize(f"{doc.title} {doc.text}"))

    assert index.document_length(doc.doc_id) == expected_len
    assert index.document_length(doc.doc_id) == 8


def test_title_is_included_in_index(index: InvertedIndex) -> None:
    assert index.term_frequency("linux", "vpn_linux") == 1
    assert index.term_frequency("windows", "vpn_windows") == 1
    assert index.term_frequency("gitlab", "gitlab_2fa") == 1


def test_term_frequency_is_correct(index: InvertedIndex) -> None:
    assert index.term_frequency("vpn", "vpn_linux") == 2
    assert index.term_frequency("vpn", "vpn_windows") == 2
    assert index.term_frequency("vpn", "gitlab_2fa") == 0


def test_document_frequency_is_correct(index: InvertedIndex) -> None:
    assert index.document_frequency("vpn") == 2
    assert index.document_frequency("на") == 2
    assert index.document_frequency("gitlab") == 1
    assert index.document_frequency("python") == 0


def test_postings_contain_document_frequencies(index: InvertedIndex) -> None:
    assert dict(index.postings_for("vpn")) == {
        "vpn_linux": 2,
        "vpn_windows": 2,
    }


def test_unknown_term_has_empty_postings(index: InvertedIndex) -> None:
    assert dict(index.postings_for("nonexistent")) == {}


def test_unknown_term_has_zero_document_frequence(index: InvertedIndex) -> None:
    assert index.document_frequency("nonexistent") == 0


def test_unknown_document_is_rejected(index: InvertedIndex) -> None:
    with pytest.raises(KeyError, match="unknown document id"):
        index.document_length("unknown")

    with pytest.raises(KeyError, match="unknown document id"):
        index.term_frequency("vpn", "unknown")


def test_duplicate_document_id_is_rejected() -> None:
    documents = [
        Document(
            doc_id="duplicate",
            title="Первый документ",
            text="Первый текст",
            category="first",
        ),
        Document(
            doc_id="duplicate",
            title="Второй документ",
            text="Второй текст",
            category="second",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="duplicate document id",
    ):
        InvertedIndex.build(documents)


def test_document_without_tokens_is_rejected() -> None:
    documents = [
        Document(
            doc_id="broken_doc",
            title="---",
            text="!!!",
            category="network",
        ),
    ]

    with pytest.raises(
        ValueError,
        match=(
            r"document 'broken_doc' "
            r"contains no indexable tokens"
        ),
    ):
        InvertedIndex.build(documents)


def test_term_is_case_normalized(index: InvertedIndex) -> None:
    assert index.document_frequency("VPN") == 2
    assert index.term_frequency("VPN", "vpn_linux") == 2

    assert dict(index.postings_for("VPN")) == {
        "vpn_linux": 2,
        "vpn_windows": 2,
    }


@pytest.mark.parametrize(
    "term",
    [
        "",
        "corporate vpn",
        "VPN-найстройка",
        "---",
    ],
)
def test_multi_token_or_empty_term_is_rejected(index: InvertedIndex, term: str) -> None:
    with pytest.raises(ValueError, match="term must contain exactly one token"):
        index.document_frequency(term)


def test_avg_document_length_is_correct(index: InvertedIndex) -> None:
    # Doc lengths in the fixture corpus are: 8, 7, 6 tokens
    assert index.statistics.avg_doc_len == pytest.approx(7.0)


def test_document_ids_are_available(index: InvertedIndex) -> None:
    assert index.document_ids == (
        "vpn_linux",
        "vpn_windows",
        "gitlab_2fa",
    )


def test_document_ids_cannot_modify_index(index: InvertedIndex) -> None:
    document_ids = index.document_ids
    modified_document_ids = document_ids + ("fake_document",)

    assert isinstance(document_ids, tuple)
    assert "fake_document" in modified_document_ids
    assert "fake_document" not in document_ids


def test_terms_are_available(index: InvertedIndex) -> None:
    assert "vpn" in index.terms
    assert "gitlab" in index.terms
    assert isinstance(index.terms, tuple)


def test_empty_document_id_is_rejected() -> None:
    documents = [
        Document(
            doc_id="   ",
            title="VPN",
            text="Настройка VPN",
            category="network",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="document id must be non-empty",
    ):
        InvertedIndex.build(documents)
