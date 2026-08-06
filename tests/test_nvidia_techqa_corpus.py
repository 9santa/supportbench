from supportbench.corpus.nvidia_techqa import _build_anomaly_report, _SourceQuery


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
