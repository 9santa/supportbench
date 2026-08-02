# NVIDIA TechQA retrieval and reranking

Final retrieval-phase record for chunk configuration `ha384o64m512r2v2`.
Full per-query outputs remain under `artifacts/evaluations/nvidia_techqa`.

## Corpus

- Parent documents: 28,481
- Runtime chunks: 165,623
- Mean chunks per parent: 5.82
- Train: 450 labeled queries
- Dev: 160 labeled queries
- Dense model: `intfloat/multilingual-e5-base`
- Cross-encoder: `BAAI/bge-reranker-v2-m3`

## Candidate recall

Initial chunk WRRF used BM25 1.0, Dense 1.0, `rrf_k=20`, and inspected 1,000 chunks.

| Ranking | R@20 | R@50 | R@100 | R@200 |
|---|---:|---:|---:|---:|
| Raw chunk positions | 0.7600 | 0.8111 | 0.8600 | 0.9067 |
| Unique parent documents | 0.7733 | 0.8333 | 0.8822 | 0.9289 |

## Parent WRRF

Selected on train: capped top-2 evidence per source, BM25 1.0, Dense 1.5,
`rrf_k=10`, source candidate depth 500.

| Split | R@20 | R@50 | R@100 | R@200 | MRR@20 |
|---|---:|---:|---:|---:|---:|
| Train | 0.7911 | 0.8489 | 0.8889 | 0.9378 | 0.5507 |
| Dev | 0.8375 | 0.9125 | 0.9313 | 0.9500 | 0.5666 |

## Parent reranking

Dev evaluation uses 20 parent candidates and two representative chunks per parent.
The independent parent score is the maximum cross-encoder chunk score. Fusion combines the
candidate ranking with the independent cross-encoder ranking using WRRF, candidate weight 1.25,
cross-encoder weight 1.0, and `rrf_k=10`.

| System | R@1 | R@3 | R@5 | R@10 | R@20 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|
| Parent WRRF candidate | 0.4688 | 0.6438 | 0.7000 | 0.7500 | 0.8375 | 0.5610 |
| Independent reranker | 0.4563 | 0.5875 | 0.6938 | **0.7875** | 0.8375 | 0.5581 |
| Candidate + reranker fusion | **0.4875** | **0.6750** | **0.7250** | 0.7813 | 0.8375 | **0.5850** |

The fusion variant is the final retrieval baseline. It gives the strongest early precision and
MRR while preserving the candidate pool recall.

## Interrupted experiment

`candidate_pool_evidence_grid_v1` was intentionally stopped during candidate generation on
2026-08-02. Its manifest is retained with status `interrupted_during_candidate_generation`; it
must not be interpreted as a completed grid result.
