# Chunk-aware RAG context

The NVIDIA TechQA context path uses the frozen retrieval baseline:

1. Run Parent WRRF over BM25 and Dense chunk rankings once.
2. Save candidate parents and representative chunks in a query-scoped run.
3. Build the independent BGE cross-encoder parent ranking from those saved chunks.
4. Fuse the saved candidate and cross-encoder rankings.
5. For the final four parents, rerank all chunks inside each parent and select top-2 evidence.
6. Materialize those chunks from the same run and pack them within the token budget.

`ParentRetrievalRun` retains the candidate parents, representative chunk IDs, reranked parents,
and fused parents. No global cache is involved: the object belongs to one query and is discarded
after context construction. The JSON CLI output includes these rankings for diagnostics.

## Current architecture

The current implementation keeps `Pipeline` for the complete query-to-answer path only:

```text
RAGPipeline
  -> ContextPreparationService.prepare(query)
     -> ParentRetrievalService.retrieve(query)
     -> RepresentativeChunkResolver.resolve(retrieval_run)
     -> RepresentativeChunkContextBuilder.build(chunks)
  -> GroundedAnswerGenerator.generate(query, context)
  -> RAGRun
```

The composition root is `supportbench.applications.nvidia_techqa`. Use
`build_nvidia_techqa_context_service(...)` for context-only diagnostics and
`build_nvidia_techqa_rag(...)` for the complete online path. Historical RAG implementations keep
their original names under `supportbench.experiments` so their experiment entry points remain
reproducible.

The model sees compact `S1`, `S2`, ... source handles instead of internal parent or chunk IDs.
After generation, handles are resolved to parent document IDs. Every included source also retains
its chunk ID, document title, section path, ordinal, original source span, and the span actually
included after overlap removal in provenance. Retrieval scores and internal IDs are deliberately
not shown to the LLM.

## Overlap removal

Representative chunks are ordered by source ordinal inside each parent. When character offsets
are available, an already-covered prefix is removed exactly. Fixed-token chunks do not have
offsets, so the builder falls back to matching the previous token suffix against the next token
prefix. The fallback requires at least eight equal tokens by default to avoid removing short,
coincidental repetitions.

## Token budget

The knowledge-context cap includes all `[DOCUMENT]` metadata, content, closing
markers, and the `[TRUNCATED]` marker. The query-specific budget additionally reserves room for
the system prompt, query, Gemma/Ollama chat template, and generated answer. If the next full chunk
does not fit, the builder uses binary search to find the largest token prefix whose fully formatted
representation fits. This is needed because decode and re-tokenization do not always preserve
token counts arithmetically. Lower-ranked parents are then omitted. Both knowledge context and the
complete prompt are counted with the generation model's tokenizer.

## Frozen local profile

The final NVIDIA TechQA profile uses four parents, two within-parent cross-encoder-selected chunks
per parent, a 4,096-token knowledge cap, an 8,192-token model window, and 1,024 reserved output
tokens. Dense retrieval defaults to CUDA; the reranker defaults to CPU so it can coexist with
Ollama Gemma on an 8 GB GPU. A larger GPU or separate retrieval service can override this with
`--reranker-device cuda`.

The application explicitly uses the measured `legacy_system_user` Ollama layout. A paired
160-query A/B found that the formally native Gemma single-user layout was faster but reduced
gold-cited answers from 95 to 85. Both layouts have prompt budgets verified against Ollama's
reported `prompt_eval_count`.

## Build a context

```bash
env -u all_proxy HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m scripts.nvidia_techqa.build_context \
  "Which Socket Gateway should I use with Netcool OMNIbus?" \
  --top-parents 4 \
  --max-context-tokens 4096 \
  --output artifacts/contexts/nvidia_techqa/socket_gateway.json
```

The JSON output contains the formatted context, parent-level citation documents, exact token
count, truncation state, and per-chunk provenance. Existing output files are preserved by default;
pass `--overwrite` explicitly to replace one.

## Generate an answer

The online generation command uses the same query-scoped retrieval and context service, then
sends the context to Ollama. The model must return the strict `answer`/`abstain`/`clarify` JSON
schema. Raw source handles are resolved to parent document IDs and validated against documents
present in the packed context.

```bash
env -u all_proxy HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m scripts.nvidia_techqa.answer \
  "Which Socket Gateway should I use with Netcool OMNIbus?" \
  --llm-model gemma3:4b \
  --top-parents 4 \
  --max-context-tokens 4096 \
  --output artifacts/answers/nvidia_techqa/socket_gateway.json
```

The answer artifact includes all candidate/reranker/fusion diagnostics, the packed context and
chunk provenance, prompt messages, raw model response, parsed answer, and validated parent
citations. When retrieval produces an empty context, the pipeline abstains without calling the
LLM.
