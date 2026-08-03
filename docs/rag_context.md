# Chunk-aware RAG context

The NVIDIA TechQA context path uses the frozen retrieval baseline:

1. Run Parent WRRF over BM25 and Dense chunk rankings once.
2. Save the top parents and two representative chunks per parent in a query-scoped run.
3. Build the independent BGE cross-encoder parent ranking from those saved chunks.
4. Fuse the saved candidate and cross-encoder rankings.
5. Materialize context chunks from the same run and pack them within the token budget.

`ParentRetrievalRun` retains the candidate parents, representative chunk IDs, reranked parents,
and fused parents. No global cache is involved: the object belongs to one query and is discarded
after context construction. The JSON CLI output includes these rankings for diagnostics.

The context cites parent document IDs. Every included source also retains its chunk ID, document
title, section path, ordinal, original source span, and the span actually included after overlap
removal. Retrieval scores are deliberately not shown to the LLM.

## Overlap removal

Representative chunks are ordered by source ordinal inside each parent. When character offsets
are available, an already-covered prefix is removed exactly. Fixed-token chunks do not have
offsets, so the builder falls back to matching the previous token suffix against the next token
prefix. The fallback requires at least eight equal tokens by default to avoid removing short,
coincidental repetitions.

## Token budget

The knowledge-context cap includes all `[DOCUMENT]` and `[CHUNK]` metadata, content, closing
markers, and the `[TRUNCATED]` marker. The query-specific budget additionally reserves room for
the system prompt, query, Gemma/Ollama chat template, and generated answer. If the next full chunk
does not fit, the builder uses the largest token prefix that does. Lower-ranked parents are then
omitted. Both knowledge context and the complete prompt are counted with the generation model's
tokenizer.

## Build a context

```bash
env -u all_proxy HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m scripts.nvidia_techqa.build_context \
  "Which Socket Gateway should I use with Netcool OMNIbus?" \
  --top-parents 5 \
  --max-context-tokens 4096 \
  --output artifacts/contexts/nvidia_techqa/socket_gateway.json
```

The JSON output contains the formatted context, parent-level citation documents, exact token
count, truncation state, and per-chunk provenance. Existing output files are preserved by default;
pass `--overwrite` explicitly to replace one.

## Generate an answer

The online generation command uses the same query-scoped retrieval and context pipeline, then
sends the context to Ollama. The model must return the strict `answer`/`abstain`/`clarify` JSON
schema. Answer citations are validated against parent document IDs present in the packed context.

```bash
env -u all_proxy HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m scripts.answer_nvidia_techqa \
  "Which Socket Gateway should I use with Netcool OMNIbus?" \
  --llm-model gemma3:4b \
  --top-parents 5 \
  --max-context-tokens 4096 \
  --output artifacts/answers/nvidia_techqa/socket_gateway.json
```

The answer artifact includes all candidate/reranker/fusion diagnostics, the packed context and
chunk provenance, prompt messages, raw model response, parsed answer, and validated parent
citations. When retrieval produces an empty context, the pipeline abstains without calling the
LLM.
