import csv
import json
from pathlib import Path

from supportbench.data.models import QueryExample
from supportbench.evaluation.retrieval_evaluator import evaluate_retriever
from supportbench.experiments.synthetic_v2.rrf_grid_config import RRFGridDefinition
from supportbench.experiments.synthetic_v2.rrf_grid_export import (
    RRFGridExperimentMetadata,
    export_rrf_grid_search,
)
from supportbench.experiments.synthetic_v2.rrf_grid_search import (
    RetrievalMetrics,
    compare_with_dense,
    run_rrf_grid_search,
)
from supportbench.retrieval.base import SearchResult


class RankingRetriever:
    def __init__(self, rankings: dict[str, tuple[str, ...]]) -> None:
        self._rankings = rankings
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        self.calls.append((query, top_k))

        return [
            SearchResult(
                doc_id=doc_id,
                score=1.0 / rank,
                rank=rank,
            )
            for rank, doc_id in enumerate(self._rankings[query][:top_k], start=1)
        ]


def test_grid_search_caches_rankings_and_applies_weights() -> None:
    query = QueryExample(
        query_id="query_1",
        query="reset access",
        relevant_doc_ids=("zzz_relevant",),
        split="dev",
    )
    bm25 = RankingRetriever(
        {
            query.query: (
                "aaa_bm25",
                "bm25_2",
                "bm25_3",
                "bm25_4",
                "bm25_5",
                "bm25_6",
                "bm25_7",
                "bm25_8",
                "bm25_9",
                "bm25_10",
            )
        }
    )
    dense = RankingRetriever(
        {
            query.query: (
                "zzz_relevant",
                "dense_2",
                "dense_3",
                "dense_4",
                "dense_5",
                "dense_6",
                "dense_7",
                "dense_8",
                "dense_9",
                "dense_10",
            )
        }
    )
    definition = RRFGridDefinition(
        bm25_weight=1.0,
        dense_weights=(1.0, 2.0),
        rrf_k_values=(10,),
        candidate_k_values=(1,),
        final_top_k=10,
    )

    result = run_rrf_grid_search(
        queries=[query],
        bm25=bm25,
        dense=dense,
        definition=definition,
    )

    assert bm25.calls == [(query.query, 50)]
    assert dense.calls == [(query.query, 50)]
    assert len(result.runs) == 2
    assert [run.evaluation.mrr for run in result.runs] == [0.5, 1.0]
    assert result.best_standalone.config.dense_weight == 2.0
    assert result.best_candidate.config.dense_weight == 2.0
    assert result.pareto_config_names == frozenset({result.runs[1].config.name})


def test_exports_grid_results_and_dense_comparison(tmp_path: Path) -> None:
    query = QueryExample(
        query_id="query_1",
        query="find target",
        relevant_doc_ids=("target_10", "target_20", "target_50"),
        split="dev",
    )
    bm25 = RankingRetriever({query.query: tuple(f"bm25_{rank}" for rank in range(1, 51))})
    dense_ranking = [f"dense_{rank}" for rank in range(1, 51)]
    dense_ranking[9] = "target_10"
    dense_ranking[19] = "target_20"
    dense_ranking[49] = "target_50"
    dense = RankingRetriever({query.query: tuple(dense_ranking)})
    definition = RRFGridDefinition(
        bm25_weight=1.0,
        dense_weights=(1.0,),
        rrf_k_values=(10,),
        candidate_k_values=(5,),
        final_top_k=10,
    )
    result = run_rrf_grid_search(
        queries=[query],
        bm25=bm25,
        dense=dense,
        definition=definition,
    )

    assert dense.calls == [(query.query, 50)]
    assert result.dense_baseline.recall_at_10 == 1.0 / 3.0
    assert result.runs[0].metrics.recall_at_20 == 0.0
    assert result.runs[0].metrics.recall_at_50 == 0.0
    assert result.runs[0].comparison_vs_dense.dense_only_hit_at_10 == 1

    export_rrf_grid_search(
        result,
        output_directory=tmp_path,
        metadata=RRFGridExperimentMetadata(
            split="dev",
            documents_path=Path("data/synthetic/v2/documents.jsonl"),
            queries_path=Path("data/synthetic/v2/queries_dev.jsonl"),
            dense_index_path=Path("artifacts/synthetic/v2/dense/multilingual-e5-base"),
            dense_model_name="test-model",
            bm25_k1=0.5,
            bm25_b=1.0,
        ),
    )

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    with (tmp_path / "grid_results.csv").open(encoding="utf-8", newline="") as file:
        csv_rows = list(csv.DictReader(file))

    assert summary["configuration_count"] == 1
    assert summary["experiment"]["query_count"] == 1
    assert summary["experiment"]["queries_path"] == (
        "data/synthetic/v2/queries_dev.jsonl"
    )
    assert summary["decision"]["recommendation"] == "keep_dense_and_skip_rrf"
    assert csv_rows[0]["recall_at_20"] == "0.0"
    assert csv_rows[0]["recall_at_50"] == "0.0"
    assert csv_rows[0]["dense_only_hit_at_10"] == "1"
    assert csv_rows[0]["relevant_documents_lost_at_50"] == "3"
    assert (tmp_path / "grid_results.jsonl").exists()
    assert (tmp_path / "finalists" / "bm25.jsonl").exists()
    assert (tmp_path / "finalists" / "dense.jsonl").exists()
    assert (tmp_path / "finalists" / "best_standalone.jsonl").exists()
    assert (tmp_path / "finalists" / "best_candidate.jsonl").exists()


def test_counts_additional_relevant_documents_when_both_retrievers_hit() -> None:
    query = QueryExample(
        query_id="query_1",
        query="multi-label query",
        relevant_doc_ids=("relevant_a", "relevant_b", "relevant_c"),
        split="dev",
    )
    dense = RankingRetriever(
        {
            query.query: (
                "relevant_a",
                "dense_noise_2",
                "dense_noise_3",
                "dense_noise_4",
                "dense_noise_5",
                "dense_noise_6",
                "dense_noise_7",
                "dense_noise_8",
                "dense_noise_9",
                "dense_noise_10",
            )
        }
    )
    hybrid = RankingRetriever(
        {
            query.query: (
                "relevant_a",
                "relevant_b",
                "hybrid_noise_3",
                "hybrid_noise_4",
                "hybrid_noise_5",
                "hybrid_noise_6",
                "hybrid_noise_7",
                "hybrid_noise_8",
                "hybrid_noise_9",
                "hybrid_noise_10",
            )
        }
    )

    comparison = compare_with_dense(
        hybrid=evaluate_retriever(
            hybrid,
            [query],
            top_k=10,
            recall_cutoffs=(1, 3, 5, 10),
        ),
        dense=evaluate_retriever(
            dense,
            [query],
            top_k=10,
            recall_cutoffs=(1, 3, 5, 10),
        ),
    )
    relevant_at_10 = comparison.relevant_documents_at_10

    assert comparison.hybrid_only_hit_at_10 == 0
    assert comparison.dense_only_hit_at_10 == 0
    assert relevant_at_10.queries_improved == 1
    assert relevant_at_10.queries_degraded == 0
    assert relevant_at_10.queries_tied == 0
    assert relevant_at_10.relevant_documents_gained == 1
    assert relevant_at_10.relevant_documents_lost == 0


def test_candidate_recall_beyond_ten_does_not_change_standalone_mrr() -> None:
    query = QueryExample(
        query_id="query_1",
        query="wide candidate query",
        relevant_doc_ids=("target_20",),
        split="dev",
    )
    ranking = [f"noise_{rank}" for rank in range(1, 51)]
    ranking[19] = "target_20"
    retriever = RankingRetriever({query.query: tuple(ranking)})

    evaluation = evaluate_retriever(retriever, [query], top_k=50)
    metrics = RetrievalMetrics.from_evaluation(evaluation)

    assert metrics.recall_at_10 == 0.0
    assert metrics.recall_at_20 == 1.0
    assert metrics.recall_at_50 == 1.0
    assert metrics.mrr == 0.0


def test_extended_metrics_include_unlabeled_queries_as_zero_recall() -> None:
    query = QueryExample(
        query_id="query_1",
        query="unsupported product query",
        relevant_doc_ids=(),
        split="dev",
    )
    retriever = RankingRetriever({query.query: tuple(f"noise_{rank}" for rank in range(1, 51))})

    evaluation = evaluate_retriever(retriever, [query], top_k=50)
    metrics = RetrievalMetrics.from_evaluation(evaluation)

    assert metrics.query_count == 1
    assert metrics.labeled_query_count == 0
    assert metrics.recall_at_20 == 0.0
    assert metrics.recall_at_50 == 0.0
