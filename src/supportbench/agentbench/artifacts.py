from pathlib import Path

from supportbench.agentbench.models import (
    AgentBenchRunConfig,
    AgentBenchSuiteMetrics,
    AgentBenchSuiteResult,
)
from supportbench.agentbench.serialization import (
    write_json_artifact,
)


def write_suite_artifacts(
    *,
    output_dir: Path,
    config: AgentBenchRunConfig,
    suite: AgentBenchSuiteResult,
    metrics: AgentBenchSuiteMetrics,
    system_prompt: str,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_json_artifact(
        path=output_dir / "config.json",
        value=config,
    )

    write_json_artifact(
        path=output_dir / "summary.json",
        value=metrics,
    )

    write_json_artifact(
        path=output_dir / "failures.json",
        value=suite.case_failures,
    )

    (output_dir / "system_prompt.txt").write_text(
        system_prompt,
        encoding="utf-8",
    )

    cases_dir = output_dir / "cases"

    for result in suite.case_results:
        write_json_artifact(
            path=(cases_dir / (result.scenario_id + ".json")),
            value=result,
        )
