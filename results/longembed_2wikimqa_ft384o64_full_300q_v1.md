# LongEmbed 2WikiMultihopQA: full_300q_v1

Completed full-corpus experiment. Full per-query JSONL and TREC runs are stored at:

`artifacts/benchmarks/longembed/2wikimqa/ft384o64/full_300q_v1`

## Reproducibility

- Dataset: `dwzhu/LongEmbed`, task `2wikimqa`
- Revision: `10039a580487dacecf79db69166e17ace3ede392`
- Queries / parent documents: 300 / 300
- Chunking: fixed-token 384, overlap 64
- Chunks: 8,766; mean 29.22 per parent; p95 54
- Chunk corpus fingerprint: `5857d50a7c8a2c92f7e675624095e2d8e75e21d2b7b833fd479b2cf07809b4b1`
- Dense: `intfloat/multilingual-e5-base`, normalized E5 query/passage embeddings
- Reranker: `BAAI/bge-reranker-v2-m3`
- Source depth / parent depth: 500 / 100
- WRRF: BM25 1.0, Dense 1.5, `rrf_k=10`
- Parent WRRF aggregation: capped top-2 chunks per source
- Independent parent reranker: max cross-encoder chunk score (`second_evidence_weight=0`)
- Fusion: candidate 1.25, cross-encoder 1.0, `rrf_k=10`
- Reranker runtime: 2,199.94 seconds on an NVIDIA GeForce RTX 3070 Ti

## Results

| System | R@1 | R@3 | R@5 | R@10 | R@20 | R@100 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| document_bm25 | 0.9333 | 0.9700 | 0.9833 | 0.9867 | 0.9900 | 1.0000 | 0.9619 |
| chunk_bm25_raw_parent | 0.9733 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9891 |
| chunk_bm25_unique_parent | 0.9733 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9891 |
| chunk_dense_raw_parent | 0.9733 | 0.9933 | 0.9933 | 0.9967 | 1.0000 | 1.0000 | 0.9860 |
| chunk_dense_unique_parent | 0.9733 | 0.9933 | 0.9933 | 0.9967 | 1.0000 | 1.0000 | 0.9866 |
| chunk_wrrf_raw_parent | 0.9933 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9975 |
| chunk_wrrf_unique_parent | 0.9933 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9975 |
| parent_wrrf | 0.9800 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9926 |
| parent_wrrf_reranked | 0.9867 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9951 |
| parent_wrrf_fused | **0.9967** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **0.9988** |

## Rank movement

Comparisons use the gold parent rank for every query, not only hit/no-hit counters.

| Comparison | Improved | Degraded | Tied |
|---|---:|---:|---:|
| parent_wrrf -> independent reranker | 6 | 4 | 290 |
| parent_wrrf -> fusion | 5 | 0 | 295 |
| chunk_wrrf_unique_parent -> fusion | 1 | 0 | 299 |

## Interpretation

Chunking materially improves document BM25 on this corpus, but 2WikiMultihopQA is nearly
saturated by chunk WRRF at R@3. It validates chunk-to-parent mechanics and reranker integration,
but is too easy to distinguish strong configurations reliably beyond rank 1. The independent
reranker improves parent WRRF, while the fusion variant has the best R@1 and nDCG@10 in this run.
