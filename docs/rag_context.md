# Chunk-aware RAG context

The NVIDIA TechQA context path uses the frozen retrieval baseline:

1. Parent WRRF over BM25 and Dense chunk rankings.
2. Two representative chunks per candidate parent.
3. Independent BGE cross-encoder parent ranking.
4. Candidate/cross-encoder rank fusion.
5. Token-budgeted context packing.

The context cites parent document IDs. Every included source also retains its chunk ID, document
title, section path, ordinal, and source character offsets. Retrieval scores are deliberately not
shown to the LLM.

## Overlap removal

Representative chunks are ordered by source ordinal inside each parent. When character offsets
are available, an already-covered prefix is removed exactly. Fixed-token chunks do not have
offsets, so the builder falls back to matching the previous token suffix against the next token
prefix. The fallback requires at least eight equal tokens by default to avoid removing short,
coincidental repetitions.

## Token budget

The budget includes all `[DOCUMENT]` and `[CHUNK]` metadata, content, closing markers, and the
`[TRUNCATED]` marker. If the next full chunk does not fit, the builder uses the largest token
prefix that does. Lower-ranked parents are then omitted. `RAGContext.token_count` is computed
from the final formatted text using the configured Hugging Face tokenizer. Set
`--context-tokenizer` to the generation model's tokenizer before enforcing its production context
limit. Until a generation model is selected, the CLI defaults to the E5 tokenizer.

## Build a context

```bash
env -u all_proxy HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m scripts.build_nvidia_techqa_context \
  "Which Socket Gateway should I use with Netcool OMNIbus?" \
  --top-parents 5 \
  --max-context-tokens 4096 \
  --output artifacts/contexts/nvidia_techqa/socket_gateway.json
```

The JSON output contains the formatted context, parent-level citation documents, exact token
count, truncation state, and per-chunk provenance. Existing output files are preserved by default;
pass `--overwrite` explicitly to replace one.
