from typing import cast

from fastapi.testclient import TestClient

from supportbench.api.agent_service import AgentRunService
from supportbench.api.app import create_app
from supportbench.api.runtime import ApiRuntime
from supportbench.api.worlds import WorldService


def _test_app_runtime() -> ApiRuntime:
    return ApiRuntime(
        world_service=cast(WorldService, object()),
        agent_run_service=cast(AgentRunService, object()),
    )


def test_demo_page_contains_supportbench_markup() -> None:
    app = create_app(runtime_factory=_test_app_runtime)

    with TestClient(app) as client:
        response = client.get("/demo")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<h1>SupportBench</h1>" in response.text
    assert "Agentic RAG over live enterprise state" in response.text
    assert 'id="world-form"' in response.text
    assert 'id="task-form"' in response.text


def test_demo_static_assets_are_reachable() -> None:
    app = create_app(runtime_factory=_test_app_runtime)

    with TestClient(app) as client:
        css = client.get("/static/demo.css")
        javascript = client.get("/static/demo.js")

    assert css.status_code == 200
    assert "text/css" in css.headers["content-type"]
    assert ".tool-trajectory" in css.text

    assert javascript.status_code == 200
    assert "javascript" in javascript.headers["content-type"]
    assert 'apiFetch("/agent/runs"' in javascript.text
    assert "/approve" in javascript.text
    assert 'document.createElement("strong")' in javascript.text
    assert "innerHTML" not in javascript.text
    assert "approval_id" not in javascript.text
