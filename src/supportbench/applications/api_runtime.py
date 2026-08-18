import os
from pathlib import Path

from supportbench.api.agent_service import AgentRunService
from supportbench.api.runs import InMemoryAgentRunStore
from supportbench.api.runtime import ApiRuntime
from supportbench.api.worlds import PostgresDemoWorldService
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
from supportbench.applications.support_agent_prompt import (
    SUPPORT_AGENT_SYSTEM_PROMPT,
)
from supportbench.llm.ollama_client import (
    OllamaToolCallingClient,
)
from supportbench.tools.policies import (
    CREATE_SUPPORT_CASE_PERMISSION,
    ENTERPRISE_READ_PERMISSION,
    SUPPORT_DOCS_READ_PERMISSION,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def build_api_runtime() -> ApiRuntime:
    database_url = _required_env("SUPPORTBENCH_SIMULATOR_DATABASE_URL")

    model_name = _resolve_agent_model()

    tokenizer_name = os.environ.get(
        "SUPPORTBENCH_MODEL_TOKENIZER",
        "Qwen/Qwen3-4B",
    ).strip()

    ollama_base_url = os.environ.get(
        "SUPPORTBENCH_OLLAMA_BASE_URL",
        "http://127.0.0.1:11434",
    ).strip()

    dense_device = os.environ.get(
        "SUPPORTBENCH_DENSE_DEVICE",
        "cuda",
    ).strip()

    reranker_device = os.environ.get(
        "SUPPORTBENCH_RERANKER_DEVICE",
        "cpu",
    ).strip()

    enterprise = build_enterprise_simulator(database_url=database_url)

    world_service = PostgresDemoWorldService(
        session_factory=enterprise.session_factory,
    )

    try:
        retrieval_config = NvidiaTechQAContextConfig(
            chunks_root=(PROJECT_ROOT / "data" / "nvidia_techqa" / "chunks"),
            index_root=(PROJECT_ROOT / "artifacts" / "nvidia_techqa" / "indexes"),
            context_tokenizer_name=tokenizer_name,
            dense_device=dense_device,
            reranker_device=reranker_device,
        )

        retrieval_runtime = build_nvidia_techqa_retrieval_runtime(retrieval_config)

        knowledge_service = build_nvidia_techqa_knowledge_service(
            retrieval_config,
            retrieval_runtime=retrieval_runtime,
        )

        model = OllamaToolCallingClient(
            model_name=model_name,
            base_url=ollama_base_url,
            temperature=0.0,
            context_window=_int_env(
                "SUPPORTBENCH_AGENT_CONTEXT_WINDOW",
                16_384,
            ),
            max_output_tokens=_int_env(
                "SUPPORTBENCH_AGENT_MAX_OUTPUT_TOKENS",
                4_096,
            ),
            think=_bool_env(
                "SUPPORTBENCH_AGENT_THINK",
                True,
            ),
        )

        support_agent = build_support_agent(
            enterprise_service=enterprise.service,
            knowledge_service=knowledge_service,
            model=model,
            max_steps=_int_env(
                "SUPPORTBENCH_AGENT_MAX_STEPS",
                8,
            ),
        )

        run_store = InMemoryAgentRunStore()

        agent_run_service = AgentRunService(
            orchestrator=support_agent.orchestrator,
            store=run_store,
            world_service=world_service,
            system_prompt=SUPPORT_AGENT_SYSTEM_PROMPT,
            default_permissions=frozenset(
                {
                    ENTERPRISE_READ_PERMISSION,
                    SUPPORT_DOCS_READ_PERMISSION,
                    CREATE_SUPPORT_CASE_PERMISSION,
                }
            ),
            actor_user_id="alice",
        )

        return ApiRuntime(
            world_service=world_service,
            agent_run_service=agent_run_service,
            close_callbacks=(
                world_service.close,
                enterprise.close,
            ),
        )

    except Exception:
        world_service.close()
        enterprise.close()
        raise


def _resolve_agent_model() -> str:
    override = os.environ.get(
        "SUPPORTBENCH_AGENT_MODEL",
        "",
    ).strip()

    if override:
        return override

    base = os.environ.get(
        "SUPPORTBENCH_MODEL",
        "",
    ).strip()

    if base:
        return base

    return "qwen3:4b"


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


def _int_env(
    name: str,
    default: int,
) -> int:
    raw = os.environ.get(
        name,
        "",
    ).strip()

    if not raw:
        return default

    value = int(raw)

    if value <= 0:
        raise ValueError(f"{name} must be positive")

    return value


def _bool_env(
    name: str,
    default: bool,
) -> bool:
    raw = (
        os.environ.get(
            name,
            "",
        )
        .strip()
        .lower()
    )

    if not raw:
        return default

    if raw in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return True

    if raw in {
        "0",
        "false",
        "no",
        "off",
    }:
        return False

    raise ValueError(f"{name} must be a boolean")
