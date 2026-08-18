import json
import os
from uuid import uuid4

from scripts._paths import PROJECT_ROOT
from supportbench.applications.enterprise_simulator import (
    build_enterprise_simulator,
)
from supportbench.applications.nvidia_techqa import (
    NvidiaTechQAContextConfig,
    build_nvidia_techqa_knowledge_service,
    build_nvidia_techqa_retrieval_runtime,
)
from supportbench.applications.support_agent import (
    build_support_agent,
)
from supportbench.llm.ollama_client import (
    OllamaToolCallingClient,
)
from supportbench.simulator.postgres.lifecycle import (
    delete_world,
    reset_world,
)
from supportbench.simulator.scenarios import (
    build_scenario,
)
from supportbench.tools.models import (
    ToolExecutionContext,
)
from supportbench.tools.policies import (
    ENTERPRISE_READ_PERMISSION,
    SUPPORT_DOCS_READ_PERMISSION,
)

SYSTEM_PROMPT = """
You are the SupportBench enterprise support agent.

Use enterprise tools for facts about the current environment,
including installed versions, service status, assets, users,
and entitlements.

Use support-document tools for technical documentation,
requirements, compatibility, known problems, fixes, and
troubleshooting guidance.

For questions that combine current environment state with
technical documentation, inspect both sources before answering.

Never invent current enterprise state.

Never claim that two product versions are compatible or
incompatible unless the returned support documentation
actually establishes the relevant requirement.

If the available documentation is insufficient to establish
compatibility, say so explicitly.

Do not create or modify anything unless the user explicitly
asks for a mutation.
"""

USER_MESSAGE = """
On asset dash-host-01, determine which DASH version is
currently installed.

Then search the support documentation for requirements relevant
to IBM Netcool/OMNIbus Web GUI 8.1 Fix Pack 7 and determine
whether the installed DASH version is sufficient.

Only make a compatibility claim if the retrieved documentation
explicitly establishes the relevant requirement. Otherwise,
state that compatibility cannot be determined from the
available evidence.
"""


def _run_retrieval_smoke_run(
    knowledge_service,
) -> None:
    query = (
        "IBM Netcool OMNIbus Web GUI 8.1 "
        "Fix Pack 7 DASH requirements "
        "Dashboard Application Services Hub"
    )

    matches = knowledge_service.search(query=query)

    print()
    print("RETRIEVAL E2E SMOKE RUN ON ONE QUERY")
    print("=" * 72)

    if not matches:
        print("no matches")
        return

    for match in matches:
        print()
        print(f"[{match.rank}] {match.document_id}")
        print(match.title)

        for chunk in match.evidence:
            section = " > ".join(chunk.section_path) if chunk.section_path else "<root>"

            print(f"  {chunk.chunk_id}")
            print(f"  section: {section}")

            preview = chunk.text[:500].replace("\n", " ")

            print(f"  {preview}")


def main() -> int:
    database_url = _get_env("SUPPORTBENCH_SIMULATOR_DATABASE_URL")

    model_name = os.environ.get(
        "SUPPORTBENCH_MODEL",
        "qwen3:4b",
    ).strip()

    tokenizer_name = os.environ.get(
        "SUPPORTBENCH_MODEL_TOKENIZER",
        "Qwen/Qwen3-4B",
    ).strip()

    ollama_base_url = os.environ.get(
        "SUPPORTBENCH_OLLAMA_BASE_URL",
        "http://127.0.0.1:11434",
    ).strip()

    world_id = f"mixed-smoke-{uuid4()}"

    request_id = f"mixed-request-{uuid4()}"

    enterprise = build_enterprise_simulator(database_url=database_url)

    try:
        retrieval_config = NvidiaTechQAContextConfig(
            chunks_root=(PROJECT_ROOT / "data" / "nvidia_techqa" / "chunks"),
            index_root=(PROJECT_ROOT / "artifacts" / "nvidia_techqa" / "indexes"),
            context_tokenizer_name=(tokenizer_name),
            dense_device="cpu",
            reranker_device="cpu",
        )

        retrieval_runtime = build_nvidia_techqa_retrieval_runtime(retrieval_config)

        knowledge_service = build_nvidia_techqa_knowledge_service(
            retrieval_config,
            retrieval_runtime=(retrieval_runtime),
        )

        _run_retrieval_smoke_run(knowledge_service)

        model = OllamaToolCallingClient(
            model_name=model_name,
            base_url=ollama_base_url,
            temperature=0.0,
            context_window=16_384,
            max_output_tokens=4_096,
            think=True,
        )

        agent = build_support_agent(
            enterprise_service=(enterprise.service),
            knowledge_service=(knowledge_service),
            model=model,
            max_steps=8,
        )

        reset_world(
            session_factory=(enterprise.session_factory),
            scenario=build_scenario(
                name="old_dash_version",
                world_id=world_id,
            ),
        )

        result = agent.orchestrator.run(
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": USER_MESSAGE,
                },
            ],
            context=ToolExecutionContext(
                world_id=world_id,
                actor_user_id="alice",
                request_id=request_id,
                permissions=frozenset(
                    {
                        ENTERPRISE_READ_PERMISSION,
                        SUPPORT_DOCS_READ_PERMISSION,
                    }
                ),
            ),
        )

        _print_trajectory(result)
        _validate_mixed_run(result)

        print()
        print("Mixed agent smoke passed.")

        return 0

    finally:
        delete_world(
            session_factory=(enterprise.session_factory),
            world_id=world_id,
        )

        enterprise.close()


def _print_trajectory(result) -> None:
    print()
    print("AGENT TRAJECTORY")
    print("=" * 72)

    print(f"status: {result.status}")

    for step in result.steps:
        print()
        print(f"STEP {step.step_index}")

        if step.assistant_content.strip():
            print(
                "assistant:",
                step.assistant_content,
            )

        for execution in step.tool_executions:
            print()
            print(f"tool: {execution.call.name}")

            print("arguments:")
            print(
                json.dumps(
                    dict(execution.call.arguments),
                    indent=2,
                    ensure_ascii=False,
                )
            )

            print("result:")

            _print_tool_result(
                execution.call.name,
                execution.result,
            )

    print()
    print("FINAL ANSWER")
    print("=" * 72)
    print(result.final_answer)


def _print_tool_result(
    tool_name: str,
    result,
) -> None:
    if result.status == "error":
        assert result.error is not None

        print(
            json.dumps(
                {
                    "status": "error",
                    "code": result.error.code,
                    "message": (result.error.message),
                },
                indent=2,
                ensure_ascii=False,
            )
        )

        return

    assert result.data is not None

    if tool_name != "search_support_docs":
        print(
            json.dumps(
                dict(result.data),
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )

        return

    matches = result.data.get("matches", [])

    preview = []

    if isinstance(matches, list):
        for match in matches:
            if not isinstance(
                match,
                dict,
            ):
                continue

            evidence_preview = []

            evidence = match.get(
                "evidence",
                [],
            )

            if isinstance(evidence, list):
                for chunk in evidence:
                    if not isinstance(
                        chunk,
                        dict,
                    ):
                        continue

                    text = str(
                        chunk.get(
                            "text",
                            "",
                        )
                    )

                    evidence_preview.append(
                        {
                            "chunk_id": (chunk.get("chunk_id")),
                            "section": (chunk.get("section")),
                            "text": (text[:500]),
                        }
                    )

            preview.append(
                {
                    "document_id": (match.get("document_id")),
                    "title": (match.get("title")),
                    "rank": (match.get("rank")),
                    "evidence": (evidence_preview),
                }
            )

    print(
        json.dumps(
            {
                "status": "success",
                "matches": preview,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _validate_mixed_run(
    result,
) -> None:
    if result.status != "completed":
        raise RuntimeError(f"mixed agent did not complete: {result.status}")

    if result.final_answer is None or not result.final_answer.strip():
        raise RuntimeError("mixed agent produced no final answer")

    executions = [execution for step in result.steps for execution in step.tool_executions]

    tool_names = [execution.call.name for execution in executions]

    if "get_installed_product" not in tool_names:
        raise RuntimeError("agent did not inspect the installed product")

    if "search_support_docs" not in tool_names:
        raise RuntimeError("agent did not search support documentation")

    if "create_support_case" in tool_names:
        raise RuntimeError("read-only mixed request attempted a mutation")

    failed = [execution for execution in executions if (execution.result.status != "success")]

    if failed:
        raise RuntimeError(
            "one or more tool executions "
            "failed: " + ", ".join(execution.call.name for execution in failed)
        )

    installed = next(
        execution for execution in executions if (execution.call.name == "get_installed_product")
    )

    assert installed.result.data is not None

    installed_version = installed.result.data.get("version")

    if installed_version != "3.1.0.3":
        raise RuntimeError(f"unexpected installed DASH version: {installed_version!r}")

    docs_search = next(
        execution for execution in executions if (execution.call.name == "search_support_docs")
    )

    assert docs_search.result.data is not None

    matches = docs_search.result.data.get("matches")

    if not isinstance(matches, list) or not matches:
        raise RuntimeError("support-document search returned no evidence")

    if "3.1.0.3" not in result.final_answer:
        raise RuntimeError("final answer omitted the observed installed version")


def _get_env(name: str) -> str:
    value = os.environ.get(
        name,
        "",
    ).strip()

    if not value:
        raise RuntimeError(f"{name} is not set")

    return value


if __name__ == "__main__":
    raise SystemExit(main())
