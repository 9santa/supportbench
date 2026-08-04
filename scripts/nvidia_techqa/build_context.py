import argparse

from scripts.nvidia_techqa._context_cli import (
    add_context_arguments,
    parent_context_payload,
    parse_context_config,
    save_json,
    validate_output_path,
)
from supportbench.applications.nvidia_techqa import build_nvidia_techqa_context_service


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a token-budgeted RAG context from fused parent retrieval."
    )
    add_context_arguments(parser)
    args = parser.parse_args()
    config = parse_context_config(parser, args)
    validate_output_path(parser, args)

    try:
        run = build_nvidia_techqa_context_service(config).prepare(args.query)
    except ValueError as error:
        parser.error(str(error))

    context = run.context
    print(f"Query: {args.query}")
    print("Retrieval: parent WRRF -> independent cross-encoder -> fusion")
    print(f"Retrieved parents: {len({chunk.parent_doc_id for chunk in run.retrieved_chunks})}")
    print(f"Representative chunks: {len(run.retrieved_chunks)}")
    print(f"Context parents: {len(context.documents)}")
    print(f"Context chunks: {len(context.provenance)}")
    context_budget = (
        run.prompt_budget.available_context_tokens
        if run.prompt_budget is not None
        else config.max_context_tokens
    )
    print(f"Context tokens: {context.token_count:,} / {context_budget:,}")
    print(
        "Full prompt tokens: "
        f"{run.prompt_token_count + config.reserved_output_tokens:,} / "
        f"{config.model_context_window:,} including output reserve"
    )
    print(f"Truncated: {str(context.truncated).lower()}")
    print()
    print(context.formatted_text)

    if args.output is not None:
        save_json(
            args.output,
            parent_context_payload(query=args.query, config=config, run=run),
        )
        print()
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
