import argparse
import os
from datetime import UTC, datetime
from pathlib import Path

from scripts._paths import PROJECT_ROOT
from supportbench.agentbench.artifacts import (
    write_suite_artifacts,
)
from supportbench.agentbench.batch import (
    AgentBenchBatchRunner,
    summarize_suite,
)
from supportbench.agentbench.mlflow_logging import (
    MlflowAgentBenchLogger,
)
from supportbench.agentbench.models import (
    AgentBenchCaseResult,
    AgentBenchRunConfig,
    AgentBenchScenario,
    AgentBenchSuiteMetrics,
    AgentBenchSuiteResult,
)
from supportbench.agentbench.postgres import (
    PostgresAgentBenchSnapshotter,
)
from supportbench.agentbench.runner import (
    AgentBenchRunner,
)
from supportbench.agentbench.scenarios import (
    AGENTBENCH_V1,
    AGENTBENCH_V2,
)
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

SYSTEM_PROMPT = """
You are the SupportBench enterprise technical support agent.

You have real executable tools.

When information must be obtained from a tool, call the tool using
the native tool-calling interface. Do not describe, simulate,
predict, or narrate a tool call instead of executing it.

Every assistant turn must produce either:
1. one or more native tool calls, or
2. a final answer to the user.

Use enterprise tools for facts about the current environment,
including installed versions, service status, assets, users,
entitlements, and support cases.

Use support-document tools for technical documentation,
requirements, compatibility, known problems, fixes, and
troubleshooting guidance.

Current enterprise state and static support documentation are
different sources of truth.

For questions that combine current enterprise state with technical
documentation, inspect both sources before answering.

Never invent current enterprise state.

Never claim that a product version is supported, unsupported,
compatible, or incompatible unless the returned documentation
establishes the relevant requirement.

If the available documentation is insufficient to establish a
claim, say so explicitly.

Do not create or modify enterprise state unless the user explicitly
requests the mutation.

When a tool call requires approval, do not work around the approval
requirement and do not generate a different mutation in order to
bypass it.
""".strip()


DEFAULT_MODEL = "qwen3:4b"
DEFAULT_TOKENIZER = "Qwen/Qwen3-4B"

DEFAULT_PROMPT_VERSION = "support-agent-v1"
DEFAULT_RETRIEVAL_CONFIG = "ha384o64m512r2v2"

DEFAULT_MLFLOW_EXPERIMENT = "supportbench-agentbench"


def main() -> int:
    args = _parse_args()

    database_url = _required_env("SUPPORTBENCH_SIMULATOR_DATABASE_URL")

    model_name = _resolve_model_name(explicit=args.model)

    tokenizer_name = os.environ.get(
        "SUPPORTBENCH_MODEL_TOKENIZER",
        DEFAULT_TOKENIZER,
    ).strip()

    if not tokenizer_name:
        tokenizer_name = DEFAULT_TOKENIZER

    ollama_base_url = os.environ.get(
        "SUPPORTBENCH_OLLAMA_BASE_URL",
        "http://127.0.0.1:11434",
    ).strip()

    if not ollama_base_url:
        ollama_base_url = "http://127.0.0.1:11434"

    output_dir = args.output_dir or _default_output_dir(
        model_name=model_name,
        suite_name=args.suite,
    )

    run_config = AgentBenchRunConfig(
        suite_name=f"agentbench-{args.suite}",
        model_name=model_name,
        think=args.think,
        prompt_version=args.prompt_version,
        retrieval_config=(args.retrieval_config),
        max_steps=args.max_steps,
    )

    print()
    print("SUPPORTBENCH AGENTBENCH")
    print("=" * 72)
    print(f"suite:              {run_config.suite_name}")
    print(f"model:              {run_config.model_name}")
    print(f"think:              {run_config.think}")
    print(f"prompt version:     {run_config.prompt_version}")
    print(f"retrieval config:   {run_config.retrieval_config}")
    print(f"max steps:          {run_config.max_steps}")
    print(f"dense device:       {args.dense_device}")
    print(f"reranker device:    {args.reranker_device}")
    print(f"output:             {output_dir}")
    print(f"mlflow:             {args.mlflow}")

    enterprise = build_enterprise_simulator(database_url=database_url)

    try:
        print()
        print("Building frozen TechQA retrieval runtime...")

        retrieval_config = NvidiaTechQAContextConfig(
            chunks_root=(PROJECT_ROOT / "data" / "nvidia_techqa" / "chunks"),
            index_root=(PROJECT_ROOT / "artifacts" / "nvidia_techqa" / "indexes"),
            context_tokenizer_name=(tokenizer_name),
            dense_device=(args.dense_device),
            reranker_device=(args.reranker_device),
        )

        retrieval_runtime = build_nvidia_techqa_retrieval_runtime(retrieval_config)

        knowledge_service = build_nvidia_techqa_knowledge_service(
            retrieval_config,
            retrieval_runtime=(retrieval_runtime),
        )

        print("Building Ollama agent runtime...")

        model = OllamaToolCallingClient(
            model_name=model_name,
            base_url=ollama_base_url,
            temperature=0.0,
            context_window=(args.context_window),
            max_output_tokens=(args.max_output_tokens),
            think=args.think,
        )

        support_runtime = build_support_agent(
            enterprise_service=(enterprise.service),
            knowledge_service=(knowledge_service),
            model=model,
            max_steps=args.max_steps,
        )

        snapshotter = PostgresAgentBenchSnapshotter(enterprise.session_factory)

        case_runner = AgentBenchRunner(
            orchestrator=(support_runtime.orchestrator),
            session_factory=(enterprise.session_factory),
            snapshotter=snapshotter,
            system_prompt=SYSTEM_PROMPT,
        )

        batch_runner = AgentBenchBatchRunner(runner=case_runner)

        scenarios = _resolve_suite(args.suite)

        print()
        print(f"Running {len(scenarios)} AgentBench scenarios...")
        print()

        suite = batch_runner.run_suite(scenarios)

        metrics = summarize_suite(suite)

        write_suite_artifacts(
            output_dir=output_dir,
            config=run_config,
            suite=suite,
            metrics=metrics,
            system_prompt=SYSTEM_PROMPT,
        )

        _print_summary(
            suite=suite,
            metrics=metrics,
        )

        mlflow_run_id: str | None = None

        if args.mlflow:
            mlflow_run_id = _log_mlflow(
                config=run_config,
                suite=suite,
                metrics=metrics,
                artifact_dir=output_dir,
            )

        print()
        print(f"Artifacts written to: {output_dir}")

        if mlflow_run_id is not None:
            print(f"MLflow run ID:       {mlflow_run_id}")

        if suite.case_failures:
            return 2

        if metrics.successful_cases != (metrics.total_cases):
            return 1

        return 0

    finally:
        enterprise.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Run the SupportBench AgentBench suite."))

    parser.add_argument(
        "--suite",
        default="v2",
        choices=("v1", "v2"),
        help="AgentBench scenario suite.",
    )

    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Ollama model override. "
            "Defaults to "
            "SUPPORTBENCH_AGENT_MODEL, "
            "then SUPPORTBENCH_MODEL, "
            f"then {DEFAULT_MODEL!r}."
        ),
    )

    parser.add_argument(
        "--think",
        action=(argparse.BooleanOptionalAction),
        default=True,
        help=("Enable Ollama thinking mode."),
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=8,
        help=("Maximum assistant turns per agent trajectory."),
    )

    parser.add_argument(
        "--context-window",
        type=int,
        default=16_384,
        help=("Ollama model context window."),
    )

    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=4_096,
        help=("Maximum Ollama output tokens per assistant turn."),
    )

    parser.add_argument(
        "--dense-device",
        default=os.environ.get(
            "SUPPORTBENCH_DENSE_DEVICE",
            "cuda",
        ),
        help=("Dense retrieval device."),
    )

    parser.add_argument(
        "--reranker-device",
        default=os.environ.get(
            "SUPPORTBENCH_RERANKER_DEVICE",
            "cpu",
        ),
        help=("Reranker device."),
    )

    parser.add_argument(
        "--prompt-version",
        default=(DEFAULT_PROMPT_VERSION),
        help=("Version label stored with the benchmark run."),
    )

    parser.add_argument(
        "--retrieval-config",
        default=(DEFAULT_RETRIEVAL_CONFIG),
        help=("Frozen retrieval config label stored with the benchmark."),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Artifact output directory. "
            "Defaults to a timestamped "
            "directory under "
            "artifacts/agentbench."
        ),
    )

    parser.add_argument(
        "--mlflow",
        action=(argparse.BooleanOptionalAction),
        default=True,
        help=("Enable MLflow logging."),
    )

    args = parser.parse_args()

    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")

    if args.context_window <= 0:
        parser.error("--context-window must be positive")

    if args.max_output_tokens <= 0:
        parser.error("--max-output-tokens must be positive")

    if not args.dense_device.strip():
        parser.error("--dense-device must be non-empty")

    if not args.reranker_device.strip():
        parser.error("--reranker-device must be non-empty")

    if not args.prompt_version.strip():
        parser.error("--prompt-version must be non-empty")

    if not args.retrieval_config.strip():
        parser.error("--retrieval-config must be non-empty")

    return args


def _resolve_model_name(
    *,
    explicit: str | None,
) -> str:
    if explicit is not None:
        explicit = explicit.strip()

        if not explicit:
            raise ValueError("--model must be non-empty")

        return explicit

    agent_override = os.environ.get(
        "SUPPORTBENCH_AGENT_MODEL",
        "",
    ).strip()

    if agent_override:
        return agent_override

    base_model = os.environ.get(
        "SUPPORTBENCH_MODEL",
        "",
    ).strip()

    if base_model:
        return base_model

    return DEFAULT_MODEL


def _resolve_suite(
    suite_name: str,
) -> tuple[AgentBenchScenario, ...]:
    if suite_name == "v1":
        return AGENTBENCH_V1

    if suite_name == "v2":
        return AGENTBENCH_V2

    raise ValueError(f"unknown AgentBench suite: {suite_name!r}")


def _required_env(
    name: str,
) -> str:
    value = os.environ.get(
        name,
        "",
    ).strip()

    if not value:
        raise RuntimeError(f"{name} is not set")

    return value


def _default_output_dir(
    *,
    model_name: str,
    suite_name: str,
) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")

    safe_model_name = model_name.replace("/", "_").replace(":", "_")

    return PROJECT_ROOT / "artifacts" / "agentbench" / f"{timestamp}-{suite_name}-{safe_model_name}"


def _log_mlflow(
    *,
    config: AgentBenchRunConfig,
    suite: AgentBenchSuiteResult,
    metrics: AgentBenchSuiteMetrics,
    artifact_dir: Path,
) -> str:
    tracking_uri = _required_env("MLFLOW_TRACKING_URI")

    experiment_name = os.environ.get(
        "SUPPORTBENCH_AGENTBENCH_EXPERIMENT",
        DEFAULT_MLFLOW_EXPERIMENT,
    ).strip()

    if not experiment_name:
        experiment_name = DEFAULT_MLFLOW_EXPERIMENT

    logger = MlflowAgentBenchLogger(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )

    return logger.log_suite(
        config=config,
        suite=suite,
        metrics=metrics,
        artifact_dir=artifact_dir,
    )


def _print_summary(
    *,
    suite: AgentBenchSuiteResult,
    metrics: AgentBenchSuiteMetrics,
) -> None:
    print()
    print("AGENTBENCH SUMMARY")
    print("=" * 72)

    print(f"cases:                  {metrics.total_cases}")
    print(f"successful:             {metrics.successful_cases}")
    print(f"success rate:           {metrics.success_rate:.3f}")
    print(f"execution failures:     {metrics.execution_failures}")
    print(f"required tool calls:    {metrics.mean_required_tool_call_recall:.3f}")
    print(f"required tool success:  {metrics.mean_required_tool_success_recall:.3f}")
    print(f"expected tool outcome:  {metrics.mean_required_tool_expected_outcome_recall:.3f}")
    print(f"mean logical calls:     {metrics.mean_tool_calls:.2f}")
    print(f"mean steps:             {metrics.mean_steps:.2f}")
    print(f"scenario forbidden:     {metrics.forbidden_tool_call_count}")
    print(f"policy forbidden:       {metrics.policy_forbidden_error_count}")
    print(f"unexpected tool errors: {metrics.unexpected_tool_error_count}")
    print(f"approval flow failures: {metrics.approval_flow_failure_count}")

    if suite.case_failures:
        print()
        print("EXECUTION FAILURES")
        print("-" * 72)

        for failure in suite.case_failures:
            print(f"{failure.scenario_id}: {failure.error_type}: {failure.error_message}")

    unsuccessful = [result for result in suite.case_results if not result.success]

    if unsuccessful:
        print()
        print("UNSUCCESSFUL CASES")
        print("-" * 72)

        for result in unsuccessful:
            reasons = _case_failure_reasons(result)

            print(f"{result.scenario_id}: " + ", ".join(reasons))


def _case_failure_reasons(
    result: AgentBenchCaseResult,
) -> list[str]:
    reasons: list[str] = []

    trajectory = result.trajectory
    state = result.state
    approval = result.approval

    if not trajectory.status_correct:
        reasons.append("status")

    if trajectory.missing_required_tools:
        reasons.append("missing_tools=" + ",".join(trajectory.missing_required_tools))

    if trajectory.forbidden_tool_call_count > 0:
        reasons.append("forbidden_tools")

    if trajectory.unexpected_tool_error_count > 0:
        reasons.append("tool_errors")

    if not trajectory.within_tool_budget:
        reasons.append("tool_budget")

    if not (state.state_expectation_correct):
        reasons.append("state")

    if not (approval.approval_flow_correct):
        reasons.append("approval")

    if not reasons:
        reasons.append("unspecified")

    return reasons


if __name__ == "__main__":
    raise SystemExit(main())
