from pathlib import Path

import mlflow

from supportbench.agentbench.models import (
    AgentBenchRunConfig,
    AgentBenchSuiteMetrics,
    AgentBenchSuiteResult,
)


class MlflowAgentBenchLogger:
    def __init__(
        self,
        *,
        tracking_uri: str,
        experiment_name: str,
    ) -> None:
        mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)

    def log_suite(
        self,
        *,
        config: AgentBenchRunConfig,
        suite: AgentBenchSuiteResult,
        metrics: AgentBenchSuiteMetrics,
        artifact_dir: Path,
    ) -> str:
        with mlflow.start_run(run_name=f"{config.suite_name}-{config.model_name}") as parent:
            mlflow.log_params(
                {
                    "suite": (config.suite_name),
                    "model": (config.model_name),
                    "think": config.think,
                    "prompt_version": (config.prompt_version),
                    "retrieval_config": (config.retrieval_config),
                    "max_steps": (config.max_steps),
                    "scenario_count": (suite.total_count),
                }
            )

            suite_metrics = {
                "trajectory_state_success_rate": (
                    metrics.trajectory_state_success_rate
                ),
                "task_success_rate": metrics.task_success_rate,
                "trajectory_state_successful_cases": float(
                    metrics.trajectory_state_successful_cases
                ),
                "task_successful_cases": float(metrics.task_successful_cases),
                "execution_failures": float(metrics.execution_failures),
                "answer_evaluated_cases": float(metrics.answer_evaluated_cases),
                "forbidden_answer_claims": float(
                    metrics.forbidden_answer_claim_count
                ),
                "mean_required_tool_call_recall": (
                    metrics.mean_required_tool_call_recall
                ),
                "mean_required_tool_success_recall": (
                    metrics.mean_required_tool_success_recall
                ),
                "mean_required_tool_expected_outcome_recall": (
                    metrics.mean_required_tool_expected_outcome_recall
                ),
                "mean_logical_tool_calls": (metrics.mean_logical_tool_calls),
                "mean_steps": (metrics.mean_steps),
                "forbidden_tool_calls": float(metrics.forbidden_tool_call_count),
                "policy_forbidden_errors": float(metrics.policy_forbidden_error_count),
                "unexpected_tool_errors": float(metrics.unexpected_tool_error_count),
                "approval_flow_failures": float(metrics.approval_flow_failure_count),
            }
            optional_suite_metrics = {
                "mean_prompt_tokens": metrics.mean_prompt_tokens,
                "mean_output_tokens": metrics.mean_output_tokens,
                "mean_model_total_duration_ms": metrics.mean_model_total_duration_ms,
                "mean_model_generation_duration_ms": (
                    metrics.mean_model_generation_duration_ms
                ),
                "answer_success_rate": metrics.answer_success_rate,
                "mean_expected_answer_fact_recall": (
                    metrics.mean_expected_answer_fact_recall
                ),
                "mean_expected_evidence_recall": (
                    metrics.mean_expected_evidence_recall
                ),
            }
            suite_metrics.update(
                {
                    key: value
                    for key, value in optional_suite_metrics.items()
                    if value is not None
                }
            )
            mlflow.log_metrics(suite_metrics)

            self._log_case_runs(suite=suite)

            mlflow.log_artifacts(
                str(artifact_dir),
                artifact_path="agentbench",
            )

            return str(parent.info.run_id)

    def _log_case_runs(
        self,
        *,
        suite: AgentBenchSuiteResult,
    ) -> None:
        for result in suite.case_results:
            with mlflow.start_run(
                run_name=result.scenario_id,
                nested=True,
            ):
                mlflow.log_params(
                    {
                        "scenario_id": (result.scenario_id),
                        "final_status": (result.run.status),
                    }
                )

                case_metrics = {
                    "success": float(result.success),
                    "trajectory_state_success": float(
                        result.trajectory_state_success
                    ),
                    "answer_success": float(result.answer.answer_success),
                    "expected_answer_fact_recall": (
                        result.answer.expected_fact_recall
                    ),
                    "expected_evidence_recall": (
                        result.answer.expected_evidence_recall
                    ),
                    "forbidden_answer_claims": float(
                        result.answer.forbidden_claim_count
                    ),
                    "required_tool_call_recall": (
                        result.trajectory.required_tool_call_recall
                    ),
                    "required_tool_success_recall": (
                        result.trajectory.required_tool_success_recall
                    ),
                    "required_tool_expected_outcome_recall": (
                        result.trajectory.required_tool_expected_outcome_recall
                    ),
                    "logical_tool_calls": float(
                        result.trajectory.logical_tool_call_count
                    ),
                    "gateway_executions": float(
                        result.trajectory.gateway_execution_count
                    ),
                    "steps": float(result.trajectory.step_count),
                    "forbidden_tool_calls": float(
                        result.trajectory.forbidden_tool_call_count
                    ),
                    "policy_forbidden_errors": float(
                        result.trajectory.policy_forbidden_error_count
                    ),
                    "unexpected_tool_errors": float(
                        result.trajectory.unexpected_tool_error_count
                    ),
                    "approval_requests": float(
                        result.trajectory.approval_required_count
                    ),
                    "state_changed": float(result.state.state_changed),
                    "support_case_delta": float(result.state.support_case_delta),
                    "audit_event_delta": float(result.state.audit_event_delta),
                    "approval_flow_correct": float(
                        result.approval.approval_flow_correct
                    ),
                }
                optional_case_metrics = {
                    "prompt_tokens": result.trajectory.prompt_token_count,
                    "output_tokens": result.trajectory.output_token_count,
                    "model_total_duration_ms": (
                        result.trajectory.model_total_duration_ms
                    ),
                    "model_generation_duration_ms": (
                        result.trajectory.model_generation_duration_ms
                    ),
                }
                case_metrics.update(
                    {
                        key: value
                        for key, value in optional_case_metrics.items()
                        if value is not None
                    }
                )
                mlflow.log_metrics(case_metrics)

        for failure in suite.case_failures:
            with mlflow.start_run(
                run_name=(failure.scenario_id),
                nested=True,
            ):
                mlflow.log_params(
                    {
                        "scenario_id": (failure.scenario_id),
                        "error_type": (failure.error_type),
                    }
                )

                mlflow.log_metric(
                    "success",
                    0.0,
                )

                mlflow.set_tag(
                    "execution_failure",
                    "true",
                )
