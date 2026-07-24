import pytest

from supportbench.data.models import Document
from supportbench.data.corpus_statistics import compute_full_corpus_statistics
from supportbench.retrieval.inverted_index import InvertedIndex


def test_computes_document_length_statistics() -> None:
    index = _build_test_index()

    statistics = compute_full_corpus_statistics(index)
    lengths = statistics.document_lengths

    # Длины документов: 2, 4, 4.
    assert lengths.minimum == 2
    assert lengths.median == 4.0
    assert lengths.mean == pytest.approx(10 / 3)
    # 2 + 0.8 * (4-2) = 3.6
    assert lengths.p90 == pytest.approx(3.6)
    assert lengths.maximum == 4
    assert lengths.standard_deviation == pytest.approx(0.9428090416)
    assert lengths.coefficient_of_variation == pytest.approx(0.2828427125)


def test_computes_posting_frequency_statistics() -> None:
    index = _build_test_index()

    statistics = compute_full_corpus_statistics(index)
    frequencies = statistics.posting_frequencies

    # TF по postings:
    # vpn:    1, 2
    # error:  1, 1
    # extra:  1
    # gitlab: 1
    # access: 3
    assert frequencies.posting_count == 7
    assert frequencies.share_tf_1 == pytest.approx(5 / 7)
    assert frequencies.share_tf_2 == pytest.approx(1 / 7)
    assert frequencies.share_tf_3_or_more == pytest.approx(1 / 7)
    assert frequencies.mean == pytest.approx(10 / 7)
    assert frequencies.maximum == 3


def test_rejects_empty_index() -> None:
    index = InvertedIndex.build([])

    with pytest.raises(
        ValueError,
        match="cannot compute statistics for an empty index",
    ):
        compute_full_corpus_statistics(index)


def _build_test_index() -> InvertedIndex:
    documents = [
        Document(
            doc_id="d1",
            title="VPN",
            text="error",
            category="test",
        ),
        Document(
            doc_id="d2",
            title="VPN",
            text="vpn error extra",
            category="test",
        ),
        Document(
            doc_id="d3",
            title="GitLab",
            text="access access access",
            category="test",
        ),
    ]

    return InvertedIndex.build(documents)
