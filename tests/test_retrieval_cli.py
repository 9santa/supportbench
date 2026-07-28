import argparse
import json
from pathlib import Path

from supportbench.evaluation.retrieval_cli import (
    EvaluationArguments,
    add_evaluation_arguments,
    add_retriever_config_arguments,
    load_evaluation_data,
    parse_evaluation_arguments,
    parse_retriever_config,
)


def test_parses_custom_retriever_configuration() -> None:
    parser = argparse.ArgumentParser()
    add_evaluation_arguments(parser)
    add_retriever_config_arguments(parser)

    namespace = parser.parse_args(
        [
            "--dense-index",
            "custom-index",
            "--dense-model",
            "custom-model",
            "--dense-device",
            "cpu",
            "--dense-batch-size",
            "8",
            "--bm25-k1",
            "1.2",
            "--bm25-b",
            "0.75",
            "--bm25-weight",
            "2.0",
            "--dense-weight",
            "3.0",
            "--candidate-k",
            "40",
            "--rrf-k",
            "20",
        ]
    )

    evaluation = parse_evaluation_arguments(parser, namespace)
    config = parse_retriever_config(parser, namespace)

    assert evaluation.top_k == 10
    assert config.dense_index_path == Path("custom-index")
    assert config.dense_model_name == "custom-model"
    assert config.dense_device == "cpu"
    assert config.dense_batch_size == 8
    assert config.bm25_k1 == 1.2
    assert config.bm25_b == 0.75
    assert config.bm25_weight == 2.0
    assert config.dense_weight == 3.0
    assert config.candidate_k == 40
    assert config.rrf_k == 20


def test_loads_only_requested_split(tmp_path: Path) -> None:
    documents_path = tmp_path / "documents.jsonl"
    queries_path = tmp_path / "queries.jsonl"
    documents_path.write_text(
        json.dumps(
            {
                "doc_id": "doc_1",
                "title": "Document",
                "text": "Document text.",
                "category": "test",
            }
        ),
        encoding="utf-8",
    )
    queries_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "query_id": "query_dev",
                        "query": "Development query",
                        "relevant_doc_ids": ["doc_1"],
                        "split": "dev",
                    }
                ),
                json.dumps(
                    {
                        "query_id": "query_test",
                        "query": "Test query",
                        "relevant_doc_ids": ["doc_1"],
                        "split": "test",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    data = load_evaluation_data(
        EvaluationArguments(
            split="test",
            documents_path=documents_path,
            queries_path=queries_path,
            top_k=10,
            failure_k=3,
        )
    )

    assert [query.query_id for query in data.queries] == ["query_test"]
