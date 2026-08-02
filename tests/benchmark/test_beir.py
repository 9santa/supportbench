import json
from pathlib import Path

from supportbench.benchmark.beir import load_beir_dataset


def test_loads_beir_dataset_and_preserves_qrel_scores(tmp_path: Path) -> None:
    (tmp_path / "qrels").mkdir()
    (tmp_path / "corpus.jsonl").write_text(
        json.dumps({"_id": "doc-1", "title": "Title", "text": "Evidence"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "queries.jsonl").write_text(
        json.dumps({"_id": "query-1", "text": "Claim"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nquery-1\tdoc-1\t2\n",
        encoding="utf-8",
    )

    dataset = load_beir_dataset(tmp_path, name="fixture", split="test")

    assert dataset.documents[0].doc_id == "doc-1"
    assert dataset.queries[0].relevant_doc_ids == ("doc-1",)
    assert dataset.qrels["query-1"]["doc-1"] == 2
