from pathlib import Path

import mlflow

from supportbench.agentbench.models import (
    AgentBenchRunConfig,
    AgentBenchSuiteResult,
    AgentBenchSuiteMetrics,
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

            mlflow.log_metrics(
                {
                    "success_rate": (metrics.success_rate),
                    "successful_cases": float(metrics.successful_cases),
                    "execution_failures": float(metrics.execution_failures),
                    "mean_required_tool_recall": (metrics.mean_required_tool_recall),
                    "mean_logical_tool_calls": (metrics.mean_tool_calls),
                    "mean_steps": (metrics.mean_steps),
                    "forbidden_tool_calls": float(metrics.forbidden_tool_call_count),
                    "unexpected_tool_errors": float(metrics.unexpected_tool_error_count),
                    "approval_flow_failures": float(metrics.approval_flow_failure_count),
                }
            )

            self._log_case_runs(suite=suite)

            mlflow.log_artifacts(
                str(artifact_dir),
                artifact_path="agentbench",
            )

            return parent.info.run_id

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

                mlflow.log_metrics(
                    {
                        "success": float(result.success),
                        "required_tool_recall": (result.trajectory.required_tool_recall),
                        "logical_tool_calls": float(result.trajectory.logical_tool_call_count),
                        "gateway_executions": float(result.trajectory.gateway_execution_count),
                        "steps": float(result.trajectory.step_count),
                        "forbidden_tool_calls": float(result.trajectory.forbidden_tool_call_count),
                        "unexpected_tool_errors": float(
                            result.trajectory.unexpected_tool_error_count
                        ),
                        "approval_requests": float(result.trajectory.approval_required_count),
                        "state_changed": float(result.state.state_changed),
                        "support_case_delta": float(result.state.support_case_delta),
                        "audit_event_delta": float(result.state.audit_event_delta),
                        "approval_flow_correct": float(result.approval.approval_flow_correct),
                    }
                )

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
