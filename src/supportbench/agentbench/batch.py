from collections.abc import Iterable, Sequence

from supportbench.agentbench.models import (
    AgentBenchCaseFailure,
    AgentBenchScenario,
    AgentBenchSuiteMetrics,
    AgentBenchSuiteResult,
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
            trajectory_state_successful_cases=0,
            task_successful_cases=0,
            execution_failures=len(suite.case_failures),
            trajectory_state_success_rate=0.0,
            task_success_rate=0.0,
            answer_evaluated_cases=0,
            answer_success_rate=None,
            mean_expected_answer_fact_recall=None,
            mean_expected_evidence_recall=None,
            forbidden_answer_claim_count=0,
            mean_required_tool_call_recall=0.0,
            mean_required_tool_success_recall=0.0,
            mean_required_tool_expected_outcome_recall=0.0,
            mean_logical_tool_calls=0.0,
            mean_steps=0.0,
            mean_prompt_tokens=None,
            mean_output_tokens=None,
            mean_model_total_duration_ms=None,
            mean_model_generation_duration_ms=None,
            forbidden_tool_call_count=0,
            policy_forbidden_error_count=0,
            unexpected_tool_error_count=0,
            approval_flow_failure_count=0,
        )

    count = len(results)
    answer_results = [result for result in results if result.answer.applicable]

    return AgentBenchSuiteMetrics(
        total_cases=suite.total_count,
        trajectory_state_successful_cases=(
            suite.trajectory_state_successful_count
        ),
        task_successful_cases=suite.task_successful_count,
        execution_failures=len(suite.case_failures),
        trajectory_state_success_rate=suite.trajectory_state_success_rate,
        task_success_rate=suite.task_success_rate,
        answer_evaluated_cases=len(answer_results),
        answer_success_rate=_mean_optional(
            float(result.answer.answer_success) for result in answer_results
        ),
        mean_expected_answer_fact_recall=_mean_optional(
            result.answer.expected_fact_recall for result in answer_results
        ),
        mean_expected_evidence_recall=_mean_optional(
            result.answer.expected_evidence_recall for result in answer_results
        ),
        forbidden_answer_claim_count=sum(
            result.answer.forbidden_claim_count for result in answer_results
        ),
        mean_required_tool_call_recall=(
            sum(result.trajectory.required_tool_call_recall for result in results) / count
        ),
        mean_required_tool_success_recall=(
            sum(result.trajectory.required_tool_success_recall for result in results) / count
        ),
        mean_required_tool_expected_outcome_recall=(
            sum(
                result.trajectory.required_tool_expected_outcome_recall
                for result in results
            )
            / count
        ),
        mean_logical_tool_calls=(
            sum(result.trajectory.logical_tool_call_count for result in results) / count
        ),
        mean_steps=(sum(result.trajectory.step_count for result in results) / count),
        mean_prompt_tokens=_mean_optional(
            result.trajectory.prompt_token_count for result in results
        ),
        mean_output_tokens=_mean_optional(
            result.trajectory.output_token_count for result in results
        ),
        mean_model_total_duration_ms=_mean_optional(
            result.trajectory.model_total_duration_ms for result in results
        ),
        mean_model_generation_duration_ms=_mean_optional(
            result.trajectory.model_generation_duration_ms for result in results
        ),
        forbidden_tool_call_count=sum(
            result.trajectory.forbidden_tool_call_count for result in results
        ),
        policy_forbidden_error_count=sum(
            result.trajectory.policy_forbidden_error_count for result in results
        ),
        unexpected_tool_error_count=sum(
            result.trajectory.unexpected_tool_error_count for result in results
        ),
        approval_flow_failure_count=sum(
            1 for result in results if not (result.approval.approval_flow_correct)
        ),
    )


def _mean_optional(values: Iterable[int | float | None]) -> float | None:
    present = [value for value in values if value is not None]

    if not present:
        return None

    return sum(present) / len(present)
