# Experiments and research process

This document is a compact map of the experimental process that led to the current SupportBench configuration. The detailed chronological notes remain under [`docs/project_history/`](project_history/).

The project deliberately kept failed hypotheses, regressions and non-selected configurations. The goal was to make the final architecture traceable to measured behavior rather than present only the winning settings.

## Experimental principles

Several rules remained stable across the project:

- compare systems on the same query set;
- preserve per-query rankings, not only aggregate scores;
- separate early-rank quality from candidate-pool coverage;
- distinguish chunk evidence from parent-document relevance;
- change one causal dimension at a time in paired RAG experiments;
- treat large metric changes as possible implementation regressions before attributing them to a model;
- preserve fingerprints/manifests for corpora and dense indexes;
- keep operational success separate from strict output-contract validity;
- for agents, score tool trajectory and resulting database state independently from final text.

## 1. Synthetic v1: lexical baselines and BM25 ablation

The first controlled corpus contained 100 short IT-support documents and 200 dev queries. It existed to establish evaluation contracts before using a real support corpus.

### Baseline

| Retriever | R@1 | R@3 | R@5 | MRR |
|---|---:|---:|---:|---:|
| TF-IDF | 0.3400 | 0.7825 | **0.9600** | 0.5907 |
| BM25 (`k1=1.5`, `b=0.75`) | 0.3525 | **0.8025** | 0.9525 | **0.6005** |

### BM25 ablation

The grid varied `k1` and `b`; the checked-in configuration is in `configs/synthetic/v1/bm25_ablation.yaml`.

The strongest early-rank profile was `k1=0.5`, `b=1.0`:

| Metric | Default BM25 | Tuned BM25 |
|---|---:|---:|
| R@1 | 0.3525 | **0.4300** |
| R@3 | **0.8025** | 0.7925 |
| R@5 | **0.9525** | 0.9375 |
| MRR | 0.6005 | **0.6488** |

The tuning improved R@1/MRR but slightly reduced broader recall. This became an early reminder that a final-ranker objective and a candidate-generator objective are not the same.

### Error analysis

Among 27 queries where both TF-IDF and BM25 missed top-3, about 81% came from two intent classes: recovery/configuration-loss and FAQ/prohibited-action questions. Retrieval usually identified the product/topic but confused document intent.

That observation motivated semantic retrieval and reranking.

Detailed history: [project_history/01_synthetic_baselines.md](project_history/01_synthetic_baselines.md).

## 2. Synthetic v1: Dense retrieval and first fusion

`intfloat/multilingual-e5-base` was introduced as the first semantic baseline.

| System | R@1 | R@3 | R@5 | MRR |
|---|---:|---:|---:|---:|
| Dense E5 | **0.7600** | **0.9600** | **0.9875** | **0.8842** |
| WRRF BM25 1 / Dense 1 | 0.6325 | 0.9125 | 0.9800 | 0.7971 |
| WRRF BM25 1 / Dense 3 | 0.7350 | 0.9450 | 0.9825 | 0.8658 |

Dense beat all tested fusion variants on the small corpus. Fusion was therefore not treated as an automatic improvement.

This changed later on TechQA, where lexical and semantic signals became more complementary.

## 3. Synthetic v2: 280-configuration WRRF grid

The second synthetic corpus used 500 dev queries (485 labeled). The WRRF grid covered:

```text
10 Dense weights
7 RRF k values
4 candidate depths
= 280 configurations
```

Configuration: `configs/synthetic/v2/rrf_grid.yaml`.

Two different profiles were intentionally retained:

- **standalone WRRF** for strong early ranking;
- **candidate WRRF** for higher candidate coverage before reranking.

| System | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.3780 | 0.6813 | 0.8263 | 0.8693 | 0.9133 | 0.9430 | 0.5770 |
| Dense | 0.6293 | **0.8343** | 0.8827 | 0.9263 | 0.9443 | 0.9563 | 0.7827 |
| Standalone WRRF 1:3, `k=10`, depth 100 | **0.6420** | 0.8323 | **0.8990** | 0.9347 | 0.9527 | 0.9633 | **0.7933** |
| Candidate WRRF 1:1.5, `k=20`, depth 100 | 0.5917 | 0.8163 | 0.8950 | **0.9370** | **0.9540** | **0.9650** | 0.7616 |

The experiment also exposed a weakness in a hit-only comparison. If both systems found at least one relevant document, a simple hit diagnostic called it a tie even when one system recovered more relevant documents. Evaluation was extended with per-cutoff relevant documents gained/lost and query improved/degraded counts.

Artifacts: `results/synthetic/v2/rrf_grid/`.

## 4. Synthetic v2: cross-encoder reranking

`BAAI/bge-reranker-v2-m3` reranked top-20 candidates independently.

| Source | Candidate R@1 | Candidate R@10 | Reranked R@1 | Reranked R@10 | Reranked MRR |
|---|---:|---:|---:|---:|---:|
| Dense | 0.6488 | 0.9550 | **0.7684** | 0.9577 | 0.8852 |
| Standalone WRRF | **0.6619** | 0.9636 | 0.7649 | 0.9625 | 0.8841 |
| Candidate WRRF | 0.6100 | **0.9660** | 0.7670 | **0.9629** | **0.8856** |

The main observation was that once the gold document was in the candidate pool, the reranker substantially reduced the differences between candidate sources.

The same stage measured latency, throughput and VRAM rather than reporting quality alone.

Artifacts: `results/synthetic/v2/reranker/`.

## 5. Why the synthetic corpus was abandoned

A first end-to-end RAG path already worked technically:

```text
retrieval -> reranking -> context -> prompt -> Gemma 3 4B -> JSON parse -> citation validation
```

But the answers merely reflected the weak, repetitive synthetic source material.

That separated two questions:

- **is the RAG mechanism grounded?** — mostly yes;
- **is the knowledge base realistic enough to make the benchmark useful?** — no.

The project therefore stopped scaling LLM-generated synthetic data and moved to NVIDIA TechQA.

## 6. TechQA document-level baselines

The normalized TechQA corpus contains 28,481 parent documents. DEV has 310 total queries, of which 160 are labeled for retrieval evaluation.

Document-level baseline:

| Retriever | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.4188 | 0.5687 | 0.6125 | 0.6813 | 0.7375 | 0.8125 | 0.4991 |
| Dense | 0.3688 | 0.5625 | 0.6000 | 0.6875 | 0.7438 | 0.8500 | 0.4784 |
| Hybrid | **0.4437** | **0.5938** | **0.6375** | **0.7312** | **0.8063** | **0.8875** | **0.5373** |

Unlike the toy corpus, BM25 and Dense were now complementary.

Detailed history: [project_history/02_techqa_and_chunking.md](project_history/02_techqa_and_chunking.md).

## 7. Retrieval regression found during chunking

After switching to a chunked corpus, BM25 recall became almost zero. The initial temptation was to blame the changed corpus, but the cause was implementation-level ranking logic.

A heap optimization combined `heapq.nlargest` with a key designed for ascending sort, effectively selecting the lowest-scoring documents. The regression was fixed by restoring the intended score-descending/doc-ID-ascending ordering.

This incident became a durable project rule: large metric shifts first trigger data/ranking invariant checks.

## 8. Fixed-token vs heading-aware chunking

### Fixed-token baselines

| Chunking | Chunks | Mean chunks/parent | R@1 | R@5 | R@10 | R@50 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ft256o32` | 135,235 | 4.75 | 0.4467 | **0.6178** | **0.6911** | **0.8156** | 0.5219 |
| `ft384o64` | 96,134 | 3.38 | 0.4533 | 0.6111 | 0.6844 | 0.8089 | 0.5230 |

`ft384o64` preserved similar quality with roughly 29% fewer chunks and became the efficiency control.

### Heading-aware v1 failure

The first structural chunker was too aggressive:

```text
chunks:               342,624
mean chunks/parent:   12.03
mean body tokens:     72.04
chunks < 50 tokens:   62.32%
```

It over-detected headings and fragmented documents.

### Heading-aware v2

Conservative heading rules plus paragraph packing produced:

```text
chunks:                  165,623
mean chunks/parent:      5.82
mean body tokens:        169.76
chunks < 50 tokens:      27.66%
chunks with section path 73.37%
formatted inputs > 512:  0
```

Retrieval comparison:

| Chunking | R@1 | R@5 | R@10 | R@50 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| `ft256o32` | 0.4467 | **0.6178** | **0.6911** | **0.8156** | 0.5219 |
| `ft384o64` | 0.4533 | 0.6111 | 0.6844 | 0.8089 | 0.5230 |
| `ha384o64m512r2v2` | **0.4689** | 0.6133 | 0.6733 | 0.8044 | **0.5344** |

Heading-aware v2 did not dominate every cutoff, but it improved early rank/MRR while preserving structure and provenance needed later by RAG. It became the frozen structural baseline.

## 9. Parent-level candidate retrieval

Chunk ranking was then converted into parent-document ranking because parent documents are the relevance/citation unit while chunks are evidence units.

A candidate recall curve showed that unique-parent coverage was already strong at larger cutoffs:

| Representation | R@20 | R@50 | R@100 | R@200 |
|---|---:|---:|---:|---:|
| Raw chunk positions | 0.7600 | 0.8111 | 0.8600 | 0.9067 |
| Unique parents | **0.7733** | **0.8333** | **0.8822** | **0.9289** |

This suggested that ranking/aggregation, not only embedding quality, was the main remaining issue.

The selected first stage used:

```text
BM25 weight          1.0
Dense weight         1.5
source RRF k         10
source depth         500
parent aggregation   capped_top_2_sum
```

Detailed history: [project_history/03_parent_retrieval_and_reranking.md](project_history/03_parent_retrieval_and_reranking.md).

## 10. Candidate + reranker fusion

On TechQA DEV:

| System | R@1 | R@3 | R@5 | R@10 | R@20 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|
| Parent WRRF candidate | 0.4688 | 0.6438 | 0.7000 | 0.7500 | 0.8375 | 0.5610 |
| Independent reranker | 0.4563 | 0.5875 | 0.6938 | **0.7875** | 0.8375 | 0.5581 |
| **Candidate + reranker fusion** | **0.4875** | **0.6750** | **0.7250** | 0.7813 | 0.8375 | **0.5850** |

Pure reranking was better at R@10 but worse at R@1/R@3/MRR. Final fusion kept both signals:

```text
candidate prior weight 1.25
cross-encoder weight   1.0
fusion RRF k           10
```

This is the frozen retrieval baseline recorded in [`results/nvidia_techqa/retrieval_reranking.md`](../results/nvidia_techqa/retrieval_reranking.md).

## 11. External validation

A separate benchmark branch tested the same architecture outside TechQA. On LongEmbed 2WikiMultihopQA, the parent-fusion path reached R@1 = 0.9967 and R@3 = 1.0000.

The point was not to claim general leaderboard superiority. It was a sanity check that the retrieval architecture itself was functioning and that TechQA's lower scores were not obviously caused by a broken parent-ranking implementation.

See [project_history/04_external_benchmarks.md](project_history/04_external_benchmarks.md).

## 12. Within-parent evidence selection

Once final parents were known, two evidence-selection strategies were paired on the same parent set:

| Selection | Gold parent in context | Exact reference in context |
|---|---:|---:|
| Retrieval representatives | 116 / 160 | 69 / 158 = 43.67% |
| Within-parent cross-encoder top-2 | 116 / 160 | **75 / 158 = 47.47%** |

The change produced 15 gains and 9 losses, net +6 exact-reference contexts. It improved evidence selection without changing parent retrieval.

## 13. Oracle context experiment: measuring context interference

The most important RAG ablation was designed to separate retrieval failure from generation interference.

Paired modes included:

```text
current
gold_injected
gold_only_selected
oracle_source
```

A key result compared two modes with the same gold evidence chunks. Removing distractor parents increased answers from 129 to 155 and strict lexical F1 from 0.2166 to 0.3223.

This showed that **inter-parent context interference** was a major failure mode. More retrieved context could make the model less useful even when the correct evidence was present.

See [rag_experiment_index.md](rag_experiment_index.md) and [project_history/06_rag_evaluation_and_freeze.md](project_history/06_rag_evaluation_and_freeze.md).

## 14. Parent-count sweep

The production ablation compared top-1 through top-5 prefixes of the same frozen parent ranking.

### Context coverage

| Parents | Gold context | Exact reference | Mean context tokens |
|---:|---:|---:|---:|
| 1 | 48.75% | 32.28% | 513 |
| 2 | 58.75% | 39.87% | 1,049 |
| 3 | 67.50% | 44.94% | 1,560 |
| 4 | 69.38% | 45.57% | 2,111 |
| 5 | 72.50% | 47.47% | 2,690 |

### Generation

| Parents | Answers | Gold-cited answers | Strict lexical F1 | Repairs | Mean generation |
|---:|---:|---:|---:|---:|---:|
| 1 | 151 | 78 | 0.2429 | 9 | 1,598 ms |
| 2 | 145 | 87 | 0.2358 | 12 | 1,726 ms |
| 3 | 138 | 92 | 0.2233 | 17 | 2,041 ms |
| **4** | **140** | **95** | 0.2174 | 20 | 2,131 ms |
| 5 | 127 | 90 | 0.2006 | 31 | 2,248 ms |

Top-1 had the highest answer rate and lexical F1, but almost half of its answers were generated without gold context. Top-4 was selected as the grounding/stability compromise. Top-5 improved coverage slightly but produced fewer answers, more repairs and worse lexical F1.

## 15. Full classic RAG DEV run

`gemma3-4b-ha-dev-v4` evaluated all 310 DEV queries:

| Metric | Result |
|---|---:|
| Operational success | 308 / 310 = 99.35% |
| Truncated | 2 / 310 |
| Answers | 256 |
| Abstentions | 52 |
| Repaired responses | 48 |
| Strict citation-contract success | 83.87% |

The evaluator itself evolved during these runs. It learned to distinguish:

- parser errors;
- truncation;
- source-handle resolution failures;
- citation-decision contract errors;
- citations outside supplied context;
- safe operational repair vs original strict validity;
- answer text that leaks source IDs or embeds a citation list.

This is why "generation success" was not treated as a single undifferentiated metric.

## 16. Gemma prompt-layout A/B

The historical classic RAG baseline compared the existing Ollama split-turn layout against Gemma's official single-user format on the same 160 top-4 contexts.

| Metric | Legacy split turns | Gemma single user |
|---|---:|---:|
| Operational success | 160 / 160 | 160 / 160 |
| Answers | **140** | 137 |
| Contract repairs | **20** | 22 |
| Strict citation contract | **87.50%** | 86.25% |
| Gold-cited answers | **95** | 85 |
| Mean generation latency | 2,278.7 ms | **2,063.8 ms** |

The native format was faster but lost ten net gold-cited answers. The frozen classic RAG profile therefore retained `legacy_system_user` as an explicitly measured choice.

See [project_history/07_gemma_prompt_layout.md](project_history/07_gemma_prompt_layout.md).

## 17. From classic RAG to a tool-using agent

The next experimental step introduced dynamic enterprise state. The key architectural choice was to avoid a nested generative RAG tool.

Instead:

```text
Qwen -> knowledge retrieval tools -> evidence -> same Qwen
Qwen -> enterprise tools          -> live state -> same Qwen
```

A first real mixed smoke demonstrated one trajectory combining current installed-product state from PostgreSQL with TechQA evidence, ending in a grounded abstention when the retrieved evidence did not establish the requested compatibility claim.

See [project_history/09_tool_gateway_and_agent.md](project_history/09_tool_gateway_and_agent.md).

## 18. AgentBench V1: benchmark-driven API discovery

The first 15-scenario agent run produced only 5/15 successes despite 96.2% required tool-call recall.

Error decomposition showed:

```text
service_not_found             13
policy forbidden errors        5
user_entitlement_not_found     3
```

The dominant problem was an application capability gap: the model could choose `get_service_status`, but there was no tool for resolving "production Web GUI" to the internal `service_id`.

The fix was not a prompt hack. A new read-only `search_services` capability was added.

## 19. AgentBench V2

On the same 15-scenario suite and the same `qwen3:4b` + thinking setup:

| Metric | V1 | V2 |
|---|---:|---:|
| Success | 33.3% | **86.7%** |
| Required tool-call recall | 96.2% | **100%** |
| Unexpected tool errors | 21 | **1** |
| Approval-flow failures | **0** | **0** |

Category result in V2:

```text
Enterprise  4/4
Knowledge   3/4
Mixed       3/4
Write       3/3
```

The remaining failures were local model/tool-selection behavior rather than another broad tool-surface gap: one policy-invalid tool selection in a documentation task and one reasoning turn that exhausted the 4096-token output budget before emitting a tool call.

Detailed methodology: [agentbench.md](agentbench.md).

## Current experiment map

The fastest way to understand why the frozen configuration looks the way it does is:

```text
synthetic lexical baselines
      ↓
BM25 ablation
      ↓
Dense baseline
      ↓
WRRF grid
      ↓
cross-encoder reranking
      ↓
real TechQA corpus
      ↓
fixed-token vs heading-aware chunking
      ↓
parent aggregation grid
      ↓
candidate + reranker fusion
      ↓
within-parent evidence A/B
      ↓
oracle context-interference experiment
      ↓
top-1...top-5 context sweep
      ↓
prompt-layout A/B
      ↓
deterministic enterprise simulator
      ↓
native tool calling
      ↓
AgentBench V1 failure decomposition
      ↓
service discovery / write-contract fix
      ↓
AgentBench V2
```

The selected settings are summarized in [frozen_configuration.md](frozen_configuration.md).
