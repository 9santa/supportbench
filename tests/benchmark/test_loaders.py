import json
from pathlib import Path

from supportbench.benchmark.loaders import (
    load_benchmark_queries,
)
from supportbench.data.loaders import (
    load_queries,
)


def _write_jsonl(
    path: Path,
    records: list[dict[str, object]],
) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_same_file_supports_legacy_and_benchmark_loaders(
    tmp_path: Path,
) -> None:
    path = tmp_path / "queries.jsonl"

    _write_jsonl(
        path,
        [
            {
                "query_id": "DEV_Q001",
                "split": "dev",
                "query": "How can I fix this?",
                "answer": "Apply the fix.",
                "is_impossible": False,
                "relevant_doc_ids": ["swg12345678"],
                "source_context_filenames": ["swg12345678.txt"],
            },
            {
                "query_id": "DEV_Q002",
                "split": "dev",
                "query": "Unsupported question",
                "answer": "-",
                "is_impossible": True,
                "relevant_doc_ids": [],
                "source_context_filenames": [],
            },
        ],
    )

    known_doc_ids = {"swg12345678"}

    legacy_queries = load_queries(
        path,
        known_doc_ids,
    )
    benchmark_queries = load_benchmark_queries(
        path,
        known_doc_ids=known_doc_ids,
    )

    assert len(legacy_queries) == 2
    assert len(benchmark_queries) == 2

    assert legacy_queries[0].query_id == (benchmark_queries[0].query_id)
    assert legacy_queries[0].query == (benchmark_queries[0].query)
    assert legacy_queries[0].relevant_doc_ids == (benchmark_queries[0].relevant_doc_ids)

    assert benchmark_queries[0].answerability == "answerable"
    assert benchmark_queries[0].reference_answer == "Apply the fix."

    assert benchmark_queries[1].answerability == "unanswerable"
    assert benchmark_queries[1].reference_answer is None
    assert benchmark_queries[1].relevant_doc_ids == ()


def test_preserves_empty_answer_as_none(
    tmp_path: Path,
) -> None:
    path = tmp_path / "queries.jsonl"

    _write_jsonl(
        path,
        [
            {
                "query_id": "DEV_Q003",
                "split": "dev",
                "query": "How can I fix this?",
                "answer": "   ",
                "is_impossible": False,
                "relevant_doc_ids": ["swg12345678"],
                "source_context_filenames": ["swg12345678.txt"],
            }
        ],
    )

    queries = load_benchmark_queries(
        path,
        known_doc_ids={"swg12345678"},
    )

    assert queries[0].answerability == ("answerable")
    assert queries[0].reference_answer is None
