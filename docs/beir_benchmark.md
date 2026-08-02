# Retrieval and reranking benchmarks

This branch has two complementary tracks and adds no benchmark-specific dependencies.

## Document-level calibration: BEIR SciFact

SciFact test has 5,183 short documents and 300 queries. It is useful for comparing scores with
published BEIR baselines, but it is intentionally evaluated without chunking: splitting these
short abstracts would not exercise SupportBench's parent-document retrieval architecture.

## Systems

- BM25
- multilingual E5 dense retrieval
- BM25 + Dense Weighted RRF
- each candidate source followed by `BAAI/bge-reranker-v2-m3`
- each candidate source fused with its cross-encoder ranking using Weighted RRF

All rerankers receive the same top-100 depth. The runner reports `nDCG`, `MAP`, `Recall`,
`Precision`, and `MRR` at standard cutoffs and writes a TREC run file for external validation.

## Run

```bash
python -m scripts.prepare_beir --dataset scifact
python -m scripts.benchmark_beir --dataset scifact
```

Every experiment writes to a unique `artifacts/benchmarks/beir/<dataset>/<experiment-name>`
directory and refuses to overwrite existing output.

## Chunking benchmark: LongEmbed 2WikiMultihopQA

The pinned `dwzhu/LongEmbed` 2WikiMultihopQA snapshot contains 300 queries and 300 genuinely
long documents. With the default E5 tokenizer and `384/64` fixed-token chunking, the corpus
produces 8,766 chunks: 29.22 chunks per parent on average and 54 at p95.

```bash
env -u all_proxy HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m scripts.prepare_longembed --task 2wikimqa
env -u all_proxy HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m scripts.benchmark_longembed_chunked --task 2wikimqa
```

The runner compares document BM25 with chunk BM25, Dense, and Weighted RRF. Chunk systems are
reported both as raw parent mappings, where duplicate parents retain their ranks, and as unique
parent rankings. It also evaluates parent-level capped-top-2 WRRF, an independent cross-encoder
parent reranker, and the legacy candidate/reranker fusion variant.

LongEmbed is downloaded from a fixed repository revision with SHA-256 verification. Chunk and
dense-index manifests retain the source corpus hash/fingerprint. Experiment outputs are stored
under `artifacts/benchmarks/longembed/<task>/<chunking-key>/<experiment-name>` and are never
overwritten.

## Sources

- BEIR paper: https://openreview.net/forum?id=wCu6T5xFjeJ
- Official repository: https://github.com/beir-cellar/beir
- Official datasets: https://github.com/beir-cellar/beir/wiki/Datasets-available
- LongEmbed paper: https://arxiv.org/abs/2404.12096
- LongEmbed repository: https://github.com/dwzhu-pku/LongEmbed
- LongEmbed dataset: https://huggingface.co/datasets/dwzhu/LongEmbed
