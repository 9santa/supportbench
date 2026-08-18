from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from supportbench.api.routes.agent import router as agent_router
from supportbench.api.routes.demo import router as demo_router
from supportbench.api.routes.health import router as health_router
from supportbench.api.routes.worlds import router as worlds_router
from supportbench.api.runtime import (
    ApiRuntime,
    RuntimeFactory,
)

_API_DIRECTORY = Path(__file__).resolve().parent


def create_app(
    *,
    runtime_factory: RuntimeFactory | None = None,
) -> FastAPI:
    resolved_factory = runtime_factory or _default_runtime_factory

    @asynccontextmanager
    async def lifespan(
        app: FastAPI,
    ) -> AsyncIterator[None]:
        runtime = resolved_factory()

        app.state.runtime = runtime

        try:
            yield
        finally:
            try:
                runtime.close()
            finally:
                app.state.runtime = None

    app = FastAPI(
        title="SupportBench",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.mount(
        "/static",
        StaticFiles(directory=_API_DIRECTORY / "static"),
        name="static",
    )

    app.include_router(demo_router)

    app.include_router(health_router)

    app.include_router(worlds_router)

    app.include_router(agent_router)

    return app


def _default_runtime_factory() -> ApiRuntime:
    from supportbench.applications.api_runtime import (
        build_api_runtime,
    )

    return build_api_runtime()


app = create_app()
