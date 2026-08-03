import argparse
import math
import sys
from dataclasses import asdict

from scripts.build_nvidia_techqa_context import (
    build_context_pipeline,
    build_parser,
    parent_context_payload,
    save_json,
    validate_arguments,
    validate_output_path,
)
from supportbench.rag.citation_validator import CitationValidationError
from supportbench.rag.generation.ollama import OllamaClientError, OllamaLLMClient
from supportbench.rag.generation.parser import GeneratedAnswerParseError
from supportbench.rag.generation.prompt import GroundedPromptBuilder
from supportbench.rag.generation.service import GroundedAnswerGenerator
from supportbench.rag.parent_pipeline import ParentContextRun, ParentGroundedRAGPipeline

DEFAULT_LLM_MODEL = "gemma3:4b"


def parse_args() -> argparse.Namespace:
    parser = build_parser(
        description="Answer a query using the chunk-aware NVIDIA TechQA RAG pipeline."
    )
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--llm-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--show-context", action="store_true")
    parser.add_argument("--show-raw-response", action="store_true")
    args = parser.parse_args()
    validate_arguments(parser, args)
    validate_output_path(parser, args)

    if not args.llm_model.strip():
        parser.error("--llm-model must be non-empty")

    if not args.ollama_url.strip():
        parser.error("--ollama-url must be non-empty")

    if args.llm_timeout_seconds <= 0.0:
        parser.error("--llm-timeout-seconds must be positive")

    if not math.isfinite(args.temperature) or args.temperature < 0.0:
        parser.error("--temperature must be finite and non-negative")

    return args


def main() -> None:
    args = parse_args()

    try:
        context_pipeline = build_context_pipeline(args)
        pipeline = ParentGroundedRAGPipeline(
            context_pipeline=context_pipeline,
            answer_generator=GroundedAnswerGenerator(
                prompt_builder=GroundedPromptBuilder(),
                llm_client=OllamaLLMClient(
                    model_name=args.llm_model,
                    base_url=args.ollama_url,
                    timeout_seconds=args.llm_timeout_seconds,
                    temperature=args.temperature,
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

    print(f"Context tokens: {run.context.token_count:,} / {args.max_context_tokens:,}")

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
        )
        payload = parent_context_payload(args, context_run)
        payload["generation"] = {
            "model": args.llm_model,
            "ollama_url": args.ollama_url,
            "temperature": args.temperature,
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
