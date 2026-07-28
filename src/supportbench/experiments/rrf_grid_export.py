import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from supportbench.experiments.evaluation_export import (
    export_query_evaluations,
)
from supportbench.experiments.rrf_grid_search import (
    GRID_EVALUATION_TOP_K,
    GRID_RECALL_CUTOFFS,
    STANDALONE_MRR_CUTOFF,
    RetrievalMetrics,
    RRFGridRun,
    RRFGridSearchResult,
)


@dataclass(frozen=True, slots=True)
class RRFGridExperimentMetadata:
    split: str
    documents_path: Path
    queries_path: Path
    dense_index_path: Path
    dense_model_name: str
    bm25_k1: float
    bm25_b: float


def export_rrf_grid_search(
    result: RRFGridSearchResult,
    *,
    output_directory: Path,
    metadata: RRFGridExperimentMetadata | None = None,
) -> None:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    pareto_names = result.pareto_config_names

    records = [
        _run_to_record(
            run,
            pareto=(run.config.name in pareto_names),
        )
        for run in result.runs
    ]

    _write_csv(
        records,
        output_directory / "grid_results.csv",
    )
    _write_jsonl(
        records,
        output_directory / "grid_results.jsonl",
    )

    best_standalone = result.best_standalone
    best_candidate = result.best_candidate

    summary = {
        "experiment": _experiment_record(result, metadata=metadata),
        "configuration_count": len(result.runs),
        "grid": asdict(result.definition),
        "bm25_baseline": _metrics_record(RetrievalMetrics.from_evaluation(result.bm25_baseline)),
        "dense_baseline": _metrics_record(RetrievalMetrics.from_evaluation(result.dense_baseline)),
        "best_standalone": _run_to_record(
            best_standalone,
            pareto=(best_standalone.config.name in pareto_names),
        ),
        "best_candidate": _run_to_record(
            best_candidate,
            pareto=(best_candidate.config.name in pareto_names),
        ),
        "pareto_config_names": sorted(pareto_names),
        "decision": _build_decision(result),
    }

    (output_directory / "summary.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    finalists_directory = output_directory / "finalists"

    export_query_evaluations(
        result.bm25_baseline,
        finalists_directory / "bm25.jsonl",
    )
    export_query_evaluations(
        result.dense_baseline,
        finalists_directory / "dense.jsonl",
    )
    export_query_evaluations(
        best_standalone.evaluation,
        finalists_directory / "best_standalone.jsonl",
    )
    export_query_evaluations(
        best_candidate.evaluation,
        finalists_directory / "best_candidate.jsonl",
    )


def _run_to_record(
    run: RRFGridRun,
    *,
    pareto: bool,
) -> dict[str, object]:
    metrics = run.metrics
    delta = run.delta_vs_dense
    comparison = run.comparison_vs_dense

    record: dict[str, object] = {
        "name": run.config.name,
        "bm25_weight": run.config.bm25_weight,
        "dense_weight": run.config.dense_weight,
        "rrf_k": run.config.rrf_k,
        "candidate_k": run.config.candidate_k,
        "final_top_k": (run.config.final_top_k),
        "query_count": metrics.query_count,
        "labeled_query_count": metrics.labeled_query_count,
        "unlabeled_query_count": metrics.query_count - metrics.labeled_query_count,
        "recall_at_1": metrics.recall_at_1,
        "recall_at_3": metrics.recall_at_3,
        "recall_at_5": metrics.recall_at_5,
        "recall_at_10": metrics.recall_at_10,
        "recall_at_20": metrics.recall_at_20,
        "recall_at_50": metrics.recall_at_50,
        "mrr": metrics.mrr,
        "mrr_cutoff": STANDALONE_MRR_CUTOFF,
        "delta_recall_at_1": (delta.recall_at_1),
        "delta_recall_at_3": (delta.recall_at_3),
        "delta_recall_at_5": (delta.recall_at_5),
        "delta_recall_at_10": (delta.recall_at_10),
        "delta_recall_at_20": delta.recall_at_20,
        "delta_recall_at_50": delta.recall_at_50,
        "delta_mrr": delta.mrr,
        "better_than_dense_by_rr": (comparison.better_by_rr),
        "worse_than_dense_by_rr": (comparison.worse_by_rr),
        "tied_with_dense_by_rr": (comparison.tied_by_rr),
        "dense_rank_1_count": (comparison.dense_rank_1_count),
        "dense_rank_1_preserved": (comparison.dense_rank_1_preserved),
        "dense_rank_1_degraded": (comparison.dense_rank_1_degraded),
        "hybrid_only_hit_at_3": (comparison.hybrid_only_hit_at_3),
        "dense_only_hit_at_3": (comparison.dense_only_hit_at_3),
        "hybrid_only_hit_at_5": (comparison.hybrid_only_hit_at_5),
        "dense_only_hit_at_5": (comparison.dense_only_hit_at_5),
        "hybrid_only_hit_at_10": (comparison.hybrid_only_hit_at_10),
        "dense_only_hit_at_10": (comparison.dense_only_hit_at_10),
        "pareto": pareto,
    }

    for k in GRID_RECALL_CUTOFFS:
        relevant_comparison = comparison.relevant_documents_at(k)
        record[f"queries_improved_at_{k}"] = relevant_comparison.queries_improved
        record[f"queries_degraded_at_{k}"] = relevant_comparison.queries_degraded
        record[f"queries_tied_at_{k}"] = relevant_comparison.queries_tied
        record[f"relevant_documents_gained_at_{k}"] = relevant_comparison.relevant_documents_gained
        record[f"relevant_documents_lost_at_{k}"] = relevant_comparison.relevant_documents_lost

    return record


def _metrics_record(
    metrics: RetrievalMetrics,
) -> dict[str, float]:
    return {
        "query_count": metrics.query_count,
        "labeled_query_count": metrics.labeled_query_count,
        "unlabeled_query_count": metrics.query_count - metrics.labeled_query_count,
        "recall_at_1": metrics.recall_at_1,
        "recall_at_3": metrics.recall_at_3,
        "recall_at_5": metrics.recall_at_5,
        "recall_at_10": metrics.recall_at_10,
        "recall_at_20": metrics.recall_at_20,
        "recall_at_50": metrics.recall_at_50,
        "mrr": metrics.mrr,
        "mrr_cutoff": STANDALONE_MRR_CUTOFF,
    }


def _write_csv(
    records: list[dict[str, object]],
    path: Path,
) -> None:
    if not records:
        raise ValueError("cannot export empty grid results")

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0]),
        )

        writer.writeheader()
        writer.writerows(records)


def _write_jsonl(
    records: list[dict[str, object]],
    path: Path,
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )
            file.write("\n")


def _build_decision(
    result: RRFGridSearchResult,
) -> dict[str, object]:
    dense = RetrievalMetrics.from_evaluation(result.dense_baseline)
    best_candidate = result.best_candidate
    best_standalone = result.best_standalone

    candidate_improves_recall_at_10 = best_candidate.metrics.recall_at_10 > dense.recall_at_10
    candidate_improves_recall_at_20 = best_candidate.metrics.recall_at_20 > dense.recall_at_20
    candidate_improves_recall_at_50 = best_candidate.metrics.recall_at_50 > dense.recall_at_50
    standalone_improves_mrr = best_standalone.metrics.mrr > dense.mrr

    candidate_comparison_at_50 = best_candidate.comparison_vs_dense.relevant_documents_at_50
    candidate_gains_relevant_documents = candidate_comparison_at_50.relevant_documents_gained > 0

    if (
        candidate_improves_recall_at_50
        or candidate_improves_recall_at_20
        or candidate_improves_recall_at_10
    ):
        recommendation = "use_rrf_for_candidate_retrieval"
    elif standalone_improves_mrr:
        recommendation = "use_rrf_as_standalone_retriever"
    elif candidate_gains_relevant_documents:
        recommendation = "inspect_rrf_candidate_coverage"
    else:
        recommendation = "keep_dense_and_skip_rrf"

    return {
        "candidate_recall_at_10_improved": candidate_improves_recall_at_10,
        "candidate_recall_at_20_improved": candidate_improves_recall_at_20,
        "candidate_recall_at_50_improved": candidate_improves_recall_at_50,
        "standalone_mrr_improved": (standalone_improves_mrr),
        "candidate_gained_relevant_documents_at_50": candidate_gains_relevant_documents,
        "recommendation": recommendation,
    }


def _experiment_record(
    result: RRFGridSearchResult,
    *,
    metadata: RRFGridExperimentMetadata | None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "query_count": result.dense_baseline.query_count,
        "recall_cutoffs": list(GRID_RECALL_CUTOFFS),
        "evaluation_top_k": GRID_EVALUATION_TOP_K,
        "mrr_cutoff": STANDALONE_MRR_CUTOFF,
    }
    labeled_query_count = sum(
        bool(query.relevant_doc_ids) for query in result.dense_baseline.queries
    )
    record["labeled_query_count"] = labeled_query_count
    record["unlabeled_query_count"] = result.dense_baseline.query_count - labeled_query_count

    if metadata is not None:
        record.update(
            {
                "split": metadata.split,
                "documents_path": str(metadata.documents_path),
                "queries_path": str(metadata.queries_path),
                "dense_index_path": str(metadata.dense_index_path),
                "dense_model_name": metadata.dense_model_name,
                "bm25_k1": metadata.bm25_k1,
                "bm25_b": metadata.bm25_b,
            }
        )

    return record
