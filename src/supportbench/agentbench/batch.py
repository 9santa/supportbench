from dataclasses import dataclass
from collections.abc import Sequence

from supportbench.agentbench.models import (
    AgentBenchCaseFailure,
    AgentBenchScenario,
    AgentBenchSuiteResult,
    AgentBenchSuiteMetrics,
)
from supportbench.agentbench.runner import AgentBenchRunner


class AgentBenchBatchRunner:
    def __init__(
        self,
        *,
        runner: AgentBenchRunner,
    ) -> None:
        self._runner = runner

    def run_suite(
        self,
        scenarios: Sequence[AgentBenchScenario],
    ) -> AgentBenchSuiteResult:
        _validate_unique_scenario_ids(scenarios)

        results = []
        failures = []

        for scenario in scenarios:
            try:
                result = self._runner.run_case(scenario)

            except Exception as exc:
                failures.append(
                    AgentBenchCaseFailure(
                        scenario_id=scenario.scenario_id,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )

                continue

            results.append(result)

        return AgentBenchSuiteResult(
            case_results=tuple(results),
            case_failures=tuple(failures),
        )


def _validate_unique_scenario_ids(
    scenarios: Sequence[AgentBenchScenario],
) -> None:
    seen: set[str] = set()

    for scenario in scenarios:
        if scenario.scenario_id in seen:
            raise ValueError(f"duplicate AgentBench scenario_id: {scenario.scenario_id}")

        seen.add(scenario.scenario_id)


def summarize_suite(
    suite: AgentBenchSuiteResult,
) -> AgentBenchSuiteMetrics:
    results = suite.case_results

    if not results:
        return AgentBenchSuiteMetrics(
            total_cases=suite.total_count,
            successful_cases=0,
            execution_failures=len(suite.case_failures),
            success_rate=0.0,
            mean_required_tool_recall=0.0,
            mean_tool_calls=0.0,
            mean_steps=0.0,
            forbidden_tool_call_count=0,
            unexpected_tool_error_count=0,
            approval_flow_failure_count=0,
        )

    count = len(results)

    return AgentBenchSuiteMetrics(
        total_cases=suite.total_count,
        successful_cases=(suite.successful_count),
        execution_failures=len(suite.case_failures),
        success_rate=suite.success_rate,
        mean_required_tool_recall=(
            sum(result.trajectory.required_tool_recall for result in results) / count
        ),
        mean_tool_calls=(sum(result.trajectory.tool_call_count for result in results) / count),
        mean_steps=(sum(result.trajectory.step_count for result in results) / count),
        forbidden_tool_call_count=sum(
            result.trajectory.forbidden_tool_call_count for result in results
        ),
        unexpected_tool_error_count=sum(
            result.trajectory.unexpected_tool_error_count for result in results
        ),
        approval_flow_failure_count=sum(
            1 for result in results if not (result.approval.approval_flow_correct)
        ),
    )
