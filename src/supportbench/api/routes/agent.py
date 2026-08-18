from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from supportbench.api.agent_service import (
    AgentRunService,
)
from supportbench.api.dependencies import (
    get_agent_run_service,
)
from supportbench.api.mappers import (
    agent_run_response,
)
from supportbench.api.models import (
    AgentRunResponse,
    CreateAgentRunRequest,
)
from supportbench.api.runs import (
    AgentRunConflictError,
    AgentRunNotFoundError,
)
from supportbench.api.worlds import (
    DemoWorldNotFoundError,
)


router = APIRouter(
    prefix="/agent/runs",
    tags=["agent"],
)


@router.post(
    "",
    response_model=AgentRunResponse,
)
def create_agent_run(
    body: CreateAgentRunRequest,
    service: AgentRunService = Depends(get_agent_run_service),
) -> AgentRunResponse:
    try:
        stored = service.create_run(
            world_id=body.world_id,
            message=body.message,
        )

    except DemoWorldNotFoundError as exc:
        raise HTTPException(
            status_code=(status.HTTP_404_NOT_FOUND),
            detail=str(exc),
        ) from exc

    return agent_run_response(stored)


@router.get(
    "/{run_id}",
    response_model=AgentRunResponse,
)
def get_agent_run(
    run_id: str,
    service: AgentRunService = Depends(get_agent_run_service),
) -> AgentRunResponse:
    try:
        stored = service.get(run_id=run_id)

    except AgentRunNotFoundError as exc:
        raise HTTPException(
            status_code=(status.HTTP_404_NOT_FOUND),
            detail=str(exc),
        ) from exc

    return agent_run_response(stored)


@router.post(
    "/{run_id}/approve",
    response_model=AgentRunResponse,
)
def approve_agent_run(
    run_id: str,
    service: AgentRunService = Depends(get_agent_run_service),
) -> AgentRunResponse:
    try:
        stored = service.approve(run_id=run_id)

    except AgentRunNotFoundError as exc:
        raise HTTPException(
            status_code=(status.HTTP_404_NOT_FOUND),
            detail=str(exc),
        ) from exc

    except AgentRunConflictError as exc:
        raise HTTPException(
            status_code=(status.HTTP_409_CONFLICT),
            detail=str(exc),
        ) from exc

    return agent_run_response(stored)
