import json
from pathlib import Path
from zipfile import ZipFile

from supportbench.corpus.nvidia_techqa import (
    _build_anomaly_report,
    _SourceQuery,
    load_nvidia_techqa_oracle_contexts,
)


def test_anomaly_report_detects_conflicting_answerability() -> None:
    report = _build_anomaly_report(
        (
            _SourceQuery(
                query_id="TRAIN_Q001",
                question="Same question?",
                answer="An answer",
                is_impossible=False,
                contexts=(),
            ),
            _SourceQuery(
                query_id="TRAIN_Q002",
                question="same question?",
                answer="",
                is_impossible=True,
                contexts=(),
            ),
            _SourceQuery(
                query_id="TRAIN_Q003",
                question="Another duplicate?",
                answer="",
                is_impossible=True,
                contexts=(),
            ),
            _SourceQuery(
                query_id="TRAIN_Q004",
                question="another duplicate?",
                answer="",
                is_impossible=True,
                contexts=(),
            ),
        )
    )

    conflicting = report["conflicting_answerability_groups"]

    assert len(conflicting) == 1
    assert conflicting[0]["query_ids"] == ["TRAIN_Q001", "TRAIN_Q002"]
    assert conflicting[0]["is_impossible"] == [False, True]


def test_loads_oracle_context_from_source_annotation(tmp_path: Path) -> None:
    dataset_zip = tmp_path / "dataset.zip"
    source_text = "Title: Source title\n\nText:\nSource body"

    with ZipFile(dataset_zip, "w") as archive:
        archive.writestr(
            "TechQA/train.json",
            json.dumps(
                [
                    {
                        "id": "DEV_Q001",
                        "question": "Question?",
                        "answer": "Source body",
                        "is_impossible": False,
                        "contexts": [
                            {
                                "filename": "source.txt",
                                "text": source_text,
                            }
                        ],
                    }
                ]
            ),
        )

    context = load_nvidia_techqa_oracle_contexts(dataset_zip)["DEV_Q001"][0]

    assert context.document_id == "source"
    assert context.title == "Source title"
    assert context.text == "Source body"
    assert context.source_text == source_text
