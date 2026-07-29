import json
from pathlib import Path
from collections.abc import Sequence

from supportbench.evaluation.retrieval_evaluator import (
    RetrievalEvaluationResult,
)
from supportbench.experiments.reranker_comparison import (
    RerankerComparisonResult,
)
from supportbench.experiments.reranker_benchmark import (
    PipelinePerformanceSummary,
)


def render_reranker_comparison(
    result: RerankerComparisonResult,
) -> str:
    sections = [
        "Reranker comparison",
        "",
        f"Queries: {result.query_count}",
        (f"Reranker candidate pool: {result.reranker_candidate_k}"),
        (f"Final result count: {result.final_top_k}"),
        "",
        _render_candidate_table(result),
        "",
        _render_reranked_table(result),
        "",
        _render_deltas(result),
    ]

    return "\n".join(sections)


def _render_candidate_table(
    result: RerankerComparisonResult,
) -> str:
    lines = [
        "Candidate source metrics:",
        "",
        (
            f"{'Source':<20}"
            f"{'R@1':>9}"
            f"{'R@3':>9}"
            f"{'R@5':>9}"
            f"{'R@10':>9}"
            f"{'R@20':>9}"
            f"{'R@50':>9}"
            f"{'MRR':>9}"
        ),
    ]

    for pipeline in result.pipelines:
        evaluation = pipeline.candidate_evaluation

        lines.append(
            f"{pipeline.name:<20}"
            f"{evaluation.recall_at_1:>9.4f}"
            f"{evaluation.recall_at_3:>9.4f}"
            f"{evaluation.recall_at_5:>9.4f}"
            f"{evaluation.recall_at_10:>9.4f}"
            f"{evaluation.recall_at_20:>9.4f}"
            f"{evaluation.recall_at_50:>9.4f}"
            f"{evaluation.mrr:>9.4f}"
        )

    return "\n".join(lines)


def _render_reranked_table(
    result: RerankerComparisonResult,
) -> str:
    lines = [
        "After cross-encoder reranking:",
        "",
        (f"{'Source':<20}{'R@1':>10}{'R@3':>10}{'R@5':>10}{'R@10':>10}{'MRR':>10}"),
    ]

    for pipeline in result.pipelines:
        evaluation = pipeline.reranked_evaluation

        lines.append(
            f"{pipeline.name:<20}"
            f"{evaluation.recall_at_1:>10.4f}"
            f"{evaluation.recall_at_3:>10.4f}"
            f"{evaluation.recall_at_5:>10.4f}"
            f"{evaluation.recall_at_10:>10.4f}"
            f"{evaluation.mrr:>10.4f}"
        )

    return "\n".join(lines)


def _render_deltas(
    result: RerankerComparisonResult,
) -> str:
    lines = [
        "Reranker deltas against each source:",
        "",
        (f"{'Source':<20}{'ΔR@1':>10}{'ΔR@3':>10}{'ΔR@5':>10}{'ΔR@10':>10}{'ΔMRR':>10}"),
    ]

    for pipeline in result.pipelines:
        candidate = pipeline.candidate_evaluation
        reranked = pipeline.reranked_evaluation

        delta_r1 = reranked.recall_at_1 - candidate.recall_at_1
        delta_r3 = reranked.recall_at_3 - candidate.recall_at_3
        delta_r5 = reranked.recall_at_5 - candidate.recall_at_5
        delta_r10 = reranked.recall_at_10 - candidate.recall_at_10
        delta_mrr = reranked.mrr - candidate.mrr

        lines.append(
            f"{pipeline.name:<20}"
            f"{delta_r1:>10.4f}"
            f"{delta_r3:>10.4f}"
            f"{delta_r5:>10.4f}"
            f"{delta_r10:>10.4f}"
            f"{delta_mrr:>10.4f}"
        )
    return "\n".join(lines)


def export_reranker_comparison(
    result: RerankerComparisonResult,
    *,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "query_count": result.query_count,
        "reranker_candidate_k": (result.reranker_candidate_k),
        "final_top_k": result.final_top_k,
        "pipelines": [
            {
                "name": pipeline.name,
                "candidate": _candidate_metrics(pipeline.candidate_evaluation),
                "reranked": _reranked_metrics(pipeline.reranked_evaluation),
            }
            for pipeline in result.pipelines
        ],
    }

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _candidate_metrics(
    result: RetrievalEvaluationResult,
) -> dict[str, int | float]:
    return {
        "query_count": result.query_count,
        "recall_at_1": result.recall_at_1,
        "recall_at_3": result.recall_at_3,
        "recall_at_5": result.recall_at_5,
        "recall_at_10": result.recall_at_10,
        "recall_at_20": result.recall_at_20,
        "recall_at_50": result.recall_at_50,
        "mrr": result.mrr,
        "mrr_cutoff": result.mrr_cutoff,
    }


def _reranked_metrics(
    result: RetrievalEvaluationResult,
) -> dict[str, int | float]:
    return {
        "query_count": result.query_count,
        "recall_at_1": result.recall_at_1,
        "recall_at_3": result.recall_at_3,
        "recall_at_5": result.recall_at_5,
        "recall_at_10": result.recall_at_10,
        "mrr": result.mrr,
        "mrr_cutoff": result.mrr_cutoff,
    }


def render_performance_table(
    summaries: Sequence[PipelinePerformanceSummary],
    *,
    gpu_name: str | None,
    gpu_power_limit_watts: float | None,
) -> str:
    lines = [
        "Performance benchmark:",
        "",
    ]

    if gpu_name is not None:
        lines.append(f"GPU: {gpu_name}")

    if gpu_power_limit_watts is not None:
        lines.append(f"GPU power limit: {gpu_power_limit_watts:g} W")
        lines.append("Manual GPU pauses: none")
        lines.append("")

    lines.extend(
        [
            (
                f"{'Source':<20}"
                f"{'Retr p50':>11}"
                f"{'Rerank p50':>12}"
                f"{'Rerank p95':>12}"
                f"{'Total p50':>11}"
                f"{'Total p95':>11}"
                f"{'Pairs/s':>11}"
            ),
        ]
    )

    for summary in summaries:
        lines.append(
            f"{summary.name:<20}"
            f"{_ms(summary.candidate_retrieval_latency.p50_seconds):>11.2f}"
            f"{_ms(summary.reranking_latency.p50_seconds):>12.2f}"
            f"{_ms(summary.reranking_latency.p95_seconds):>12.2f}"
            f"{_ms(summary.total_latency.p50_seconds):>11.2f}"
            f"{_ms(summary.total_latency.p95_seconds):>11.2f}"
            f"{summary.pairs_per_second:>11.2f}"
        )

    lines.extend(
        [
            "",
            "VRAM:",
            "",
            (
                f"{'Source':<20}"
                f"{'Peak alloc GiB':>16}"
                f"{'Peak reserve GiB':>18}"
                f"{'Rerank +GiB':>15}"
                f"{'Batches/s':>12}"
            ),
        ]
    )

    for summary in summaries:
        lines.append(
            f"{summary.name:<20}"
            f"{_gib(summary.peak_allocated_bytes):>16.3f}"
            f"{_gib(summary.peak_reserved_bytes):>18.3f}"
            f"{_gib(summary.peak_reranking_incremental_bytes):>15.3f}"
            f"{summary.batches_per_second:>12.2f}"
        )

    return "\n".join(lines)


def _ms(seconds: float) -> float:
    return seconds * 1_000.0


def _gib(byte_count: int) -> float:
    return byte_count / (1024**3)
