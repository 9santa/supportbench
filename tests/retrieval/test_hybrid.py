import pytest

from supportbench.retrieval.base import SearchResult
from supportbench.retrieval.hybrid import weighted_rrf_fusion


def test_fuses_precomputed_rankings_with_weights() -> None:
    results = weighted_rrf_fusion(
        (
            (
                (
                    SearchResult("parent_a", 0.9, 1),
                    SearchResult("parent_b", 0.8, 2),
                ),
                1.25,
            ),
            (
                (
                    SearchResult("parent_b", 0.9, 1),
                    SearchResult("parent_a", 0.8, 2),
                ),
                1.0,
            ),
        ),
        top_k=2,
        rrf_k=10,
    )

    assert [result.doc_id for result in results] == ["parent_a", "parent_b"]
    assert results[0].score == pytest.approx(1.25 / 11 + 1.0 / 12)
    assert results[1].score == pytest.approx(1.25 / 12 + 1.0 / 11)
