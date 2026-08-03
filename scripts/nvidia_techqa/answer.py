import argparse
import math
import sys
from dataclasses import asdict

from scripts.nvidia_techqa._context_cli import (
    add_context_arguments,
    parent_context_payload,
    parse_context_config,
    save_json,
    validate_output_path,
)
from supportbench.applications.nvidia_techqa import build_nvidia_techqa_context_pipeline
from supportbench.rag.citation_validator import CitationValidationError
from supportbench.rag.generation.ollama import OllamaClientError, OllamaLLMClient
from supportbench.rag.generation.parser import GeneratedAnswerParseError
from supportbench.rag.generation.prompt import GroundedPromptBuilder
from supportbench.rag.generation.service import GroundedAnswerGenerator
from supportbench.rag.parent_pipeline import ParentContextRun, ParentGroundedRAGPipeline

DEFAULT_LLM_MODEL = "gemma3:4b"


def parse_args() -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(
        description="Answer a query using the chunk-aware NVIDIA TechQA RAG pipeline."
    )
    add_context_arguments(parser)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--llm-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--show-context", action="store_true")
    parser.add_argument("--show-raw-response", action="store_true")
    args = parser.parse_args()
    validate_output_path(parser, args)

    if not args.llm_model.strip():
        parser.error("--llm-model must be non-empty")

    if not args.ollama_url.strip():
        parser.error("--ollama-url must be non-empty")

    if args.llm_timeout_seconds <= 0.0:
        parser.error("--llm-timeout-seconds must be positive")

    if not math.isfinite(args.temperature) or args.temperature < 0.0:
        parser.error("--temperature must be finite and non-negative")

    return args, parser


def main() -> None:
    args, parser = parse_args()
    config = parse_context_config(parser, args)

    try:
        context_pipeline = build_nvidia_techqa_context_pipeline(config)
        pipeline = ParentGroundedRAGPipeline(
            context_pipeline=context_pipeline,
            answer_generator=GroundedAnswerGenerator(
                prompt_builder=GroundedPromptBuilder(),
                llm_client=OllamaLLMClient(
                    model_name=args.llm_model,
                    base_url=args.ollama_url,
                    timeout_seconds=args.llm_timeout_seconds,
                    temperature=args.temperature,
                    context_window=config.model_context_window,
                    max_output_tokens=config.reserved_output_tokens,
                ),
            ),
        )
        run = pipeline.run(args.query)
    except GeneratedAnswerParseError as error:
        _print_generation_error("Generation contract error", error, args)
        raise SystemExit(2) from error
    except CitationValidationError as error:
        _print_generation_error("Citation validation error", error, args)
        raise SystemExit(3) from error
    except OllamaClientError as error:
        print(f"LLM request error: {error}", file=sys.stderr)
        raise SystemExit(4) from error
    except ValueError as error:
        print(f"RAG pipeline error: {error}", file=sys.stderr)
        raise SystemExit(5) from error

    answer = run.answer
    print(f"Decision: {answer.decision}")
    print("Answer:")
    print(answer.answer)
    print("Citations:")

    if answer.citation_ids:
        for citation_id in answer.citation_ids:
            print(f"- {citation_id}")
    else:
        print("- none")

    context_budget = (
        run.prompt_budget.available_context_tokens
        if run.prompt_budget is not None
        else config.max_context_tokens
    )
    print(f"Context tokens: {run.context.token_count:,} / {context_budget:,}")
    print(
        "Full prompt tokens: "
        f"{run.prompt_token_count + config.reserved_output_tokens:,} / "
        f"{config.model_context_window:,} including output reserve"
    )

    if args.show_context:
        print()
        print("Context:")
        print(run.context.formatted_text)

    if args.show_raw_response:
        print()
        print("Raw response:")
        print(run.raw_response if run.raw_response is not None else "<LLM was not called>")

    if args.output is not None:
        context_run = ParentContextRun(
            retrieval=run.retrieval,
            retrieved_chunks=run.retrieved_chunks,
            context=run.context,
            prompt_budget=run.prompt_budget,
            prompt_token_count=run.prompt_token_count,
        )
        payload = parent_context_payload(
            query=args.query,
            config=config,
            run=context_run,
        )
        payload["generation"] = {
            "model": args.llm_model,
            "ollama_url": args.ollama_url,
            "temperature": args.temperature,
            "model_context_window": config.model_context_window,
            "reserved_output_tokens": config.reserved_output_tokens,
            "messages": [asdict(message) for message in run.messages],
            "raw_response": run.raw_response,
            "answer": asdict(run.answer),
        }
        save_json(args.output, payload)
        print()
        print(f"Saved: {args.output}")


def _print_generation_error(
    label: str,
    error: GeneratedAnswerParseError | CitationValidationError,
    args: argparse.Namespace,
) -> None:
    print(f"{label}: {error}", file=sys.stderr)

    if not args.show_raw_response:
        return

    print()
    print("Raw response:")
    raw_response = error.raw_response
    print(raw_response if raw_response is not None else "<not available>")


if __name__ == "__main__":
    main()
