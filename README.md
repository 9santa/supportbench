# SupportBench

SupportBench is an end-to-end experimental system for evaluating enterprise technical-support RAG and tool-using agents.

The project started as a retrieval benchmark, moved from synthetic support documents to NVIDIA TechQA / IBM technotes, and then grew into a complete agentic support stack: hybrid retrieval, cross-encoder reranking, grounded RAG, a deterministic PostgreSQL enterprise simulator, a typed Tool Gateway with authorization and approval, native Ollama tool calling, AgentBench, a FastAPI JSON API, and a small browser demo.

The central question is not just *"can an LLM answer a support question?"* It is:

> Can the system retrieve the right evidence, distinguish static documentation from live enterprise state, choose and execute the right tools, respect mutation policy, survive approval/resume boundaries, and leave the database in the expected state?

## Why this project exists

A convincing support-agent evaluation needs more than a final-answer score.

Early experiments exposed several different failure classes that can't be distinguished if only the final answer text is inspected:

- retrieval can miss the relevant parent document;
- chunking can destroy useful document structure;
- reranking can improve deep recall while hurting early ranks (which might be unintuitive);
- adding more context can increase evidence coverage but make generation worse (e.g. `Lost in the Middle` paper/phenomenon);
- the model can select the correct tool but have no way to resolve an internal entity ID;
- a mutation can be requested correctly but must still be blocked until explicit approval;
- the model can claim success even when the database mutation failed.

SupportBench separates these layers and evaluates them independently.

## What is included

- **NVIDIA TechQA dataset ingestion** with a pinned Hugging Face dataset revision.
- **Lexical retrieval**: TF-IDF, BM25 and deterministic inverted-index ranking.
- **Dense retrieval**: `intfloat/multilingual-e5-base` + FAISS.
- **Hybrid retrieval** with Weighted Reciprocal Rank Fusion.
- **Fixed-token and heading-aware chunking** with provenance and source offsets.
- **Parent-level retrieval** that aggregates evidence from multiple chunks.
- **Cross-encoder reranking** with `BAAI/bge-reranker-v2-m3`.
- **Grounded classic RAG** with token-budgeted context, structured output and citation validation.
- **Deterministic PostgreSQL enterprise simulator** with seeded worlds, cases and audit events.
- **Typed Tool Gateway** with strict Pydantic schemas, fail-closed policy coverage and structured tool errors.
- **Authorization and explicit approval** for mutations.
- **Idempotent support-case creation** bound to request and tool-call identity.
- **Native Ollama tool calling** with `qwen3:4b` as the default in the current agent runtime (any LLM with ollama tool calling support should work).
- **AgentBench** for trajectory, state-transition, approval and answer/evidence checks.
- **MLflow tracking** for retrieval, RAG and agent benchmark runs.
- **FastAPI API + Jinja2/vanilla-JS demo UI**.
- **Docker Compose** for PostgreSQL, migrations, MLflow and the API runtime.

## Architecture

```text
                                      STATIC KNOWLEDGE
                               NVIDIA TechQA / IBM technotes
                                          │
                         BM25 + multilingual E5 over chunks
                                          │
                              parent aggregation + BGE rerank
                                          │
                              search/read support-doc tools
                                          │
                                          ▼
Browser ──► FastAPI ──► AgentRunService ──► qwen3:4b
   │                                      │       │
   │                                      │       ├── final grounded answer
   │                                      │       │
   │                                      │       ▼
   │                                      └── Tool Gateway
   │                                              │
   │                         ┌────────────────────┴───────────────────┐
   │                         │                                        │
   │                         ▼                                        ▼
   │                 Knowledge tools                         Enterprise tools
   │                                                                  │
   │                                                         policy / approval
   │                                                                  │
   │                                                                  ▼
   └────────────────────────────────────────────────────────── PostgreSQL simulator
                                                              dynamic enterprise state
```

The system keeps two sources of truth separate:

- **Static truth**: technical documentation retrieved from TechQA.
- **Dynamic truth**: current services, installed products, entitlements and support cases stored in PostgreSQL.

The model is allowed to synthesize across both, but it does not get trusted execution context from the user. `world_id`, actor identity, permissions, request identity and approved calls are injected from the application side.

A deeper component and trust-boundary description is in [docs/architecture.md](docs/architecture.md).

## Key engineering decisions

| Problem discovered | Decision |
|---|---|
| Synthetic documents were too template-like | Move the main benchmark to real NVIDIA TechQA / IBM support documents |
| Long technotes exceeded useful retrieval/generation granularity | Introduce chunk-aware retrieval and compare fixed-token vs heading-aware chunking |
| Raw chunk top-k over-represented long parents | Rank on parent-document relevance while keeping chunks as evidence units |
| Independent reranking improved some deep ranks but regressed early ranks | Fuse candidate ranking and cross-encoder ranking instead of replacing the candidate prior |
| More retrieved documents increased evidence coverage but also context interference | Freeze classic RAG at top-4 parents after paired top-1...top-5 generation experiments |
| AgentBench V1 showed repeated `service_not_found` errors | Add explicit `search_services` entity resolution instead of making the model guess internal IDs |
| Mutation arguments initially included user identity | Bind case creation to the trusted actor from `ToolExecutionContext` |
| A model-issued mutation must not execute immediately | Require permission + exact approval bound to world, actor, request, call ID, tool and arguments |
| Final text alone cannot prove a mutation succeeded | Score the resulting PostgreSQL state and audit-event deltas in AgentBench |

## Evaluation results

### Retrieval

The frozen TechQA retrieval baseline is evaluated on **160 labeled DEV queries** over **28,481 parent documents** and **165,623 heading-aware chunks**.

| System | R@1 | R@3 | R@5 | R@10 | R@20 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|
| Parent WRRF candidate | 0.4688 | 0.6438 | 0.7000 | 0.7500 | 0.8375 | 0.5610 |
| Independent parent reranker | 0.4563 | 0.5875 | 0.6938 | **0.7875** | 0.8375 | 0.5581 |
| **Candidate + reranker fusion** | **0.4875** | **0.6750** | **0.7250** | 0.7813 | 0.8375 | **0.5850** |

The final path keeps the candidate prior because pure reranking did not dominate early-rank metrics.

Detailed retrieval results: [results/nvidia_techqa/retrieval_reranking.md](results/nvidia_techqa/retrieval_reranking.md).

### Classic RAG baseline

The classic TechQA generation experiments used `gemma3:4b` to isolate and evaluate retrieval/context/generation behavior before the agent layer was introduced.

A full 310-query DEV run reached:

- **308 / 310 operationally successful** outputs;
- **2 / 310 truncations**;
- **256 answers** and **52 abstentions**;
- **83.87% strict citation-contract success**;
- 48 non-answer responses safely repaired while preserving a separate strict-invalid diagnostic.

Paired context experiments showed that more evidence is not automatically better. Top-4 parents produced the largest number of gold-cited answers in the top-1...top-5 sweep (95 vs 92 for top-3 and 90 for top-5), while top-5 increased repairs and context interference.

See [docs/rag_experiment_index.md](docs/rag_experiment_index.md) and [docs/project_history/06_rag_evaluation_and_freeze.md](docs/project_history/06_rag_evaluation_and_freeze.md).

### AgentBench

AgentBench runs a frozen scenario against a fresh PostgreSQL world, records the complete tool trajectory, snapshots state before/after execution, checks approval behavior and evaluates deterministic answer/evidence expectations. The current CLI defaults to suite V3, which adds deterministic final-answer facts and evidence checks while preserving the earlier task set.

The first two recorded 15-scenario runs (V1 and V2) used the same `qwen3:4b` configuration with thinking enabled:

| Metric | V1 | V2 |
|---|---:|---:|
| Scenario success | 5 / 15 (**33.3%**) | 13 / 15 (**86.7%**) |
| Required tool-call recall | 96.2% | **100%** |
| Unexpected tool errors | 21 | **1** |
| Approval-flow failures | **0** | **0** |

V1 was intentionally preserved as a baseline. Its dominant failure was not simply "the model is weak": 13 errors were `service_not_found` because the tool surface had no service-name-to-ID resolver. Adding `search_services` and tightening write identity handling removed that architectural bottleneck without weakening the approval policy.

Detailed methodology and failure analysis: [docs/agentbench.md](docs/agentbench.md).

## Two representative flows

### Mixed enterprise state + documentation

A task can ask which DASH version is installed **now** and what IBM documentation establishes about Web GUI requirements.

```text
user task
  │
  ├── enterprise tool -> current installed product from PostgreSQL
  │
  ├── knowledge tool  -> TechQA evidence
  │
  └── qwen3:4b        -> synthesis that keeps live state and documentation claims separate
```

If the retrieved documentation does not establish a compatibility claim, the expected behavior is to say that the evidence is insufficient rather than invent an answer.

### Approval-gated mutation

```text
user asks to open a support case
        │
        ▼
agent checks current service state
        │
        ▼
create_support_case(...)
        │
        ▼
approval_required              PostgreSQL unchanged
        │
        ▼
operator approves the run
        │
        ▼
execute the exact pending ToolCall
        │
        ├── support_cases +1
        └── audit_events  +1
        │
        ▼
agent continues and returns the final answer
```

The browser never receives or submits the internal approval ID. The server restores the original trusted context and grants only the exact pending approval.

## Tool surface

Enterprise tools:

- `search_services`
- `get_service_status`
- `search_products`
- `get_installed_product`
- `check_user_entitlement`
- `create_support_case` — permission-gated and approval-required

Knowledge tools:

- `search_support_docs`
- `read_support_doc`

All LLM-facing arguments use strict Pydantic schemas with `extra="forbid"`.

## Tech stack

- Python 3.14
- FastAPI + Jinja2 + vanilla JavaScript
- PostgreSQL 17 + SQLAlchemy + Alembic
- Ollama (`qwen3:4b` for the current agent runtime)
- `intfloat/multilingual-e5-base`
- FAISS
- `BAAI/bge-reranker-v2-m3`
- MLflow
- Docker Compose
- Pytest, Ruff, mypy

## Quick start

### Prerequisites

You need:

- Python 3.14;
- Docker + Docker Compose;
- Ollama installed on the host;
- enough disk space for TechQA, Hugging Face model caches and the local FAISS index.

The repository intentionally does **not** commit the source dataset, generated chunks, FAISS index or model weights.

### 1. Install the data-preparation dependencies

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dense]"
```

### 2. Prepare the pinned TechQA corpus and dense index

CPU:

```bash
python -m scripts.nvidia_techqa.prepare --dense-device cpu
```

CUDA:

```bash
python -m scripts.nvidia_techqa.prepare --dense-device cuda
```

The script downloads the pinned `nvidia/TechQA-RAG-Eval` revision, normalizes the corpus, builds the frozen heading-aware chunk corpus and creates the FAISS dense index. Re-running the command skips existing stages; use `--force` to forcefully rebuild them.

### 3. Prepare the model

```bash
ollama pull qwen3:4b
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

### 4. Configure and start the stack

```bash
cp .env.example .env
docker compose up --build
```

On first API startup the Hugging Face will also download the E5 model, BGE reranker and Qwen tokenizer used by the runtime.
This can take some time.

Open:

- Demo UI: <http://127.0.0.1:8000/demo>
- OpenAPI: <http://127.0.0.1:8000/docs>
- Health: <http://127.0.0.1:8000/health>
- MLflow: <http://127.0.0.1:5000>

### Clean reproducibility test

To verify PostgreSQL bootstrap and Alembic migrations from an empty volume:

```bash
docker compose down -v
docker compose up --build
```

`-v` deletes database and MLflow volumes; do not use it if you need to preserve local experiment state.

More details: [docs/reproducibility.md](docs/reproducibility.md).

## JSON API

The demo UI is only a thin client over the same JSON API:

```text
GET    /health
POST   /worlds
DELETE /worlds/{world_id}
POST   /agent/runs
GET    /agent/runs/{run_id}
POST   /agent/runs/{run_id}/approve
GET    /demo
```

A run is a single task trajectory rather than a persistent multi-turn chatbot session.

## Reproducing experiments

The repository contains dedicated runner scripts for retrieval, chunking, reranking, context and RAG experiments under `scripts/nvidia_techqa/`, plus AgentBench under `scripts/agentbench/`.

Useful entry points:

```bash
python -m scripts.nvidia_techqa.evaluate_retrieval --help
python -m scripts.nvidia_techqa.run_parent_rrf_grid --help
python -m scripts.nvidia_techqa.run_parent_reranker_grid --help
python -m scripts.nvidia_techqa.compare_evidence_selection --help
python -m scripts.nvidia_techqa.compare_parent_counts --help
python -m scripts.nvidia_techqa.evaluate_rag --help
python -m scripts.agentbench.run --help
```

MLflow is intentionally integrated at the experiment-runner boundary rather than inside retrieval or model components. See [docs/experiment_tracking.md](docs/experiment_tracking.md).

## Repository layout

```text
src/supportbench/
  retrieval/        lexical, dense and hybrid retrieval
  chunking/         fixed-token and heading-aware chunking
  reranking/        cross-encoder and parent reranking
  rag/              context preparation and classic grounded generation
  knowledge/        bounded retrieval-as-tools adapter
  simulator/        deterministic enterprise domain + PostgreSQL adapter
  tools/            typed Tool Gateway, handlers, policy and approval
  llm/              Ollama native tool-calling adapter
  agent/            agent orchestration and approval resume
  agentbench/       scenario runner and deterministic scoring
  api/              FastAPI JSON API + demo UI
  applications/     composition roots

scripts/
  nvidia_techqa/    preparation and experiment runners
  agentbench/       agent benchmark runner

docs/               architecture, experiments, benchmark notes and history
results/            compact checked-in benchmark summaries
```

## Documentation

Start with [docs/README.md](docs/README.md).

The main documents are:

- [Architecture](docs/architecture.md) — current system and trust boundaries.
- [Experiments](docs/experiments.md) — baselines, grids, ablations and the experimental path.
- [Benchmarks](docs/benchmarks.md) — compact retrieval, RAG and AgentBench results.
- [Frozen configuration](docs/frozen_configuration.md) — the currently selected parameters and why they were selected.
- [Reproducibility](docs/reproducibility.md) — data preparation, Docker, PostgreSQL/Alembic and MLflow.
- [Project history](docs/project_history.md) — chronological engineering/research journal.

## Limitations

- TechQA is an IBM-heavy public support corpus and does not claim to be a real private enterprise knowledge base.
- The corpus and labels represent an older support-document snapshot; benchmark results should not be interpreted as current IBM product guidance.
- The deterministic enterprise simulator is intentionally small. It exists to make tool/state evaluation reproducible, not to emulate a complete ITSM platform.
- AgentBench V1/V2 contain 15 curated scenarios, so they are a diagnostic engineering benchmark rather than a statistically broad model leaderboard.
- The current demo uses an in-memory run store; restarting the API loses suspended/completed demo-run state.
- Model generation can still be nondeterministic even at temperature 0, especially with local runtime/model-version changes.

## Main takeaways

The most useful result of the project was not a single score. It was the ability to **decompose failures**.

Retrieval experiments showed when lexical and dense signals were complementary. Chunking experiments exposed structural and ranking regressions. RAG ablations showed that additional context can hurt. AgentBench showed that a missing application capability can dominate apparent model quality. State snapshots caught cases that final-text evaluation would miss. Approval tests verified that write safety remained intact while the rest of the agent improved.

That evaluation-driven loop is the core of SupportBench.
