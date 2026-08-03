import pytest

from supportbench.experiments.synthetic_v2.rag.context_builder import (
    ContextBuilder,
)
from supportbench.rag.models import (
    RetrievedDocument,
)


def make_retrieved_document(
    doc_id: str,
    *,
    rank: int,
    text: str | None = None,
    score: float = 0.9,
) -> RetrievedDocument:
    return RetrievedDocument(
        doc_id=doc_id,
        title=f"Title {doc_id}",
        text=(text if text is not None else f"Content {doc_id}"),
        category="support",
        score=score,
        rank=rank,
    )


def test_documents_are_formatted_with_ids() -> None:
    context = ContextBuilder().build(
        [
            make_retrieved_document(
                "gitlab_2fa_recovery",
                rank=1,
            )
        ]
    )

    assert "doc_id: gitlab_2fa_recovery" in context.formatted_text


def test_title_text_and_category_are_included() -> None:
    document = RetrievedDocument(
        doc_id="doc_a",
        title="Reset GitLab 2FA",
        text="Create a support request.",
        category="access",
        score=0.91,
        rank=1,
    )

    context = ContextBuilder().build([document])

    assert "title: Reset GitLab 2FA" in context.formatted_text
    assert "category: access" in context.formatted_text
    assert "Create a support request." in context.formatted_text


def test_retrieval_scores_are_not_included() -> None:
    context = ContextBuilder().build(
        [
            make_retrieved_document(
                "doc_a",
                rank=1,
                score=0.912345,
            )
        ]
    )

    assert "score" not in context.formatted_text.lower()
    assert "0.912345" not in context.formatted_text


def test_order_is_preserved() -> None:
    context = ContextBuilder().build(
        [
            make_retrieved_document(
                "doc_a",
                rank=1,
            ),
            make_retrieved_document(
                "doc_b",
                rank=2,
            ),
        ]
    )

    first_position = context.formatted_text.index("doc_id: doc_a")
    second_position = context.formatted_text.index("doc_id: doc_b")

    assert first_position < second_position


def test_max_documents_is_applied() -> None:
    context = ContextBuilder(max_documents=2).build(
        [
            make_retrieved_document(
                "doc_a",
                rank=1,
            ),
            make_retrieved_document(
                "doc_b",
                rank=2,
            ),
            make_retrieved_document(
                "doc_c",
                rank=3,
            ),
        ]
    )

    assert [document.doc_id for document in context.documents] == [
        "doc_a",
        "doc_b",
    ]
    assert context.truncated is True


def test_text_does_not_exceed_character_budget() -> None:
    context = ContextBuilder(max_characters=160).build(
        [
            make_retrieved_document(
                "doc_a",
                rank=1,
                text="A" * 1_000,
            )
        ]
    )

    assert len(context.formatted_text) <= 160


def test_oversized_first_document_is_truncated() -> None:
    original_text = "A" * 1_000

    context = ContextBuilder(max_characters=160).build(
        [
            make_retrieved_document(
                "doc_a",
                rank=1,
                text=original_text,
            )
        ]
    )

    assert len(context.documents) == 1
    assert context.documents[0].text != original_text
    assert len(context.documents[0].text) < len(original_text)


def test_truncation_marker_is_added() -> None:
    context = ContextBuilder(max_characters=160).build(
        [
            make_retrieved_document(
                "doc_a",
                rank=1,
                text="A" * 1_000,
            )
        ]
    )

    assert "[TRUNCATED]" in context.formatted_text


def test_truncated_flag_is_set() -> None:
    context = ContextBuilder(max_characters=160).build(
        [
            make_retrieved_document(
                "doc_a",
                rank=1,
                text="A" * 1_000,
            )
        ]
    )

    assert context.truncated is True


def test_empty_documents_create_empty_context() -> None:
    context = ContextBuilder().build([])

    assert context.documents == ()
    assert context.formatted_text == ""
    assert context.truncated is False


def test_duplicate_document_ids_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="duplicate retrieved document ID",
    ):
        ContextBuilder().build(
            [
                make_retrieved_document(
                    "doc_a",
                    rank=1,
                ),
                make_retrieved_document(
                    "doc_a",
                    rank=2,
                ),
            ]
        )


def test_non_consecutive_ranks_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="ranks must be consecutive",
    ):
        ContextBuilder().build(
            [
                make_retrieved_document(
                    "doc_a",
                    rank=1,
                ),
                make_retrieved_document(
                    "doc_b",
                    rank=3,
                ),
            ]
        )


def test_invalid_max_documents_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="max_documents must be positive",
    ):
        ContextBuilder(max_documents=0)


def test_invalid_max_characters_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="max_characters must be positive",
    ):
        ContextBuilder(max_characters=0)
